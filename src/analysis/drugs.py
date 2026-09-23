from __future__ import annotations

import ast
import asyncio
import gzip
import json
import os
import random
from collections import defaultdict
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB import FastMMCIFParser
from config import config
from database import Session
from database.datamodel.models import MetalBindingSite, Peptide
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from network.network import read_network_json, write_network_json
from network.utils import tqdm_joblib
from preprocessing.api import api_request
from pydantic import BaseModel, Field
from scipy.stats import false_discovery_control  # type: ignore[import-untyped]
from sqlalchemy import select
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from sqlalchemy.orm import Session as SessionType


def get_mbss_by_species(session: SessionType, species: str) -> list[MetalBindingSite]:
    species = "%" if species == "all" else species
    entries = (
        session.execute(
            select(MetalBindingSite)
            .join(Peptide)
            .where(Peptide.species.ilike(f"{species}"))
            .distinct()
        )
        .scalars()
        .all()
    )
    return entries  # type: ignore[return-value]


def get_representative_ids(mbs_ids: list[int]) -> list[int]:
    """Get representative MBS ids from a list of MBS ids."""
    with Session() as session:
        mbss = (
            session.execute(
                select(MetalBindingSite).where(MetalBindingSite.id.in_(mbs_ids))
            )
            .scalars()
            .all()
        )

    representative_ids: list[int] = []
    for mbs in mbss:
        if mbs.representative_id is not None:
            representative_ids.append(mbs.representative_id)
        else:
            representative_ids.append(mbs.id)
    return representative_ids


def process_structure_file(
    file_path: Path,
    mbs_id: int,
    pdb_id: str,
    coordinate: tuple[float, float, float],
    threshold: float = 5.0,
) -> tuple[int, set[str]]:
    """Process a mmCIF file to extract all heteroatom residues (except water) that are within
    `threshold` Å of a given metal coordinate. Returns a tuple of the MBS id and
    a set of heteroatom residue names."""
    hetero_residues: set[str] = set()
    parser = FastMMCIFParser(QUIET=True)
    try:
        with gzip.open(file_path, "rt") as handle:
            structure = parser.get_structure(pdb_id, handle)
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] != " ":
                            if residue.get_resname() == "HOH":
                                continue

                            # Check if any atom in the residue is within `threshold` Å of the
                            # provided ligand coordinate
                            for atom in residue.get_atoms():
                                atom_coord = atom.get_coord()
                                dx = atom_coord[0] - coordinate[0]
                                dy = atom_coord[1] - coordinate[1]
                                dz = atom_coord[2] - coordinate[2]
                                thr2 = threshold * threshold
                                if dx * dx + dy * dy + dz * dz <= thr2:
                                    hetero_residues.add(residue.get_resname())
                                    break
    except Exception:
        pass

    return mbs_id, hetero_residues


def checkpoint_path() -> Path:
    ckpt_dir = config.directory.analysis / "drugs"
    snapshot = ckpt_dir / "residue_mbs_map.json"
    return snapshot


def load_snapshot() -> tuple[dict[str, set[int]], set[int]]:
    """Load full snapshot if present; derive completed mbs_ids."""
    snapshot = checkpoint_path()
    residue_to_mbss: dict[str, set[int]] = {}
    completed_ids: set[int] = set()

    if snapshot.exists():
        with snapshot.open() as f:
            data: dict[str, Any] = json.load(f)  # {residue: [mbs_ids, ...]}
        # Convert to {residue: {ids}} and union to build completed_ids
        for residue, ids in data.items():
            s = set(map(int, ids))
            residue_to_mbss[residue] = s
            completed_ids.update(s)

    return residue_to_mbss, completed_ids


def atomic_write_snapshot(residue_map: dict[str, set[int]]) -> None:
    """Write the full aggregated map to JSON atomically."""
    snapshot = checkpoint_path()
    tmp = snapshot.with_suffix(snapshot.suffix + ".tmp")
    serializable = {k: list(v) for k, v in residue_map.items()}
    with tmp.open("w") as f:
        json.dump(serializable, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(snapshot)


def task_iter_filtered(
    mbss: list[MetalBindingSite], skip_ids: set[int]
) -> Generator[tuple[Path, int, str, list[float]], Any, None]:
    for mbs in mbss:
        if mbs.id in skip_ids:
            continue
        yield (
            config.directory.structures / mbs.assembly.cif_file,
            mbs.id,
            mbs.assembly.entry_pdb_id,
            mbs.ligand_coord,
        )


def tuple_adapter(
    args: tuple[Path, int, str, tuple[float, float, float], float],
) -> tuple[int, set[str]]:
    return process_structure_file(*args)


def worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def get_heteroatom_residues(species: str) -> dict[str, set[int]]:
    """Retrieve a dictionary mapping each unique heteroatom residue name to a set
    of MBS ids where that residue is present."""
    # Load any previously saved progress.
    residue_to_mbss, completed_ids = load_snapshot()

    # Build a list of tasks for parallelization.
    with Session() as session:
        mbss = get_mbss_by_species(session, species)
        total = len(mbss)
        to_do = total - len(completed_ids)
        print(
            f"Found {total} MBSs for species '{species}' "
            f"({to_do} remaining; {len(completed_ids)} already completed)"
        )

        # Build a filtered, streaming task iterator.
        task_iter = task_iter_filtered(mbss, completed_ids)

        # Process structures in parallel.
        max_workers = os.cpu_count() - 4  # type: ignore[operator]
        checkpoint_every = 1000
        processed_since_snapshot = 0
        with Pool(
            processes=max_workers, maxtasksperchild=100, initializer=worker_init
        ) as pool:
            iterator = pool.imap_unordered(tuple_adapter, task_iter, chunksize=1)  # type: ignore[arg-type]

            pbar = tqdm(total=to_do, desc="Processing structures")
            for mbs_id, hetero_set in iterator:
                for residue in hetero_set:
                    residue_to_mbss.setdefault(residue, set()).add(mbs_id)

                processed_since_snapshot += 1
                pbar.update(1)

                if processed_since_snapshot >= checkpoint_every:
                    atomic_write_snapshot(residue_to_mbss)
                    processed_since_snapshot = 0

            pbar.close()

        # Final snapshot at the end
        if processed_since_snapshot > 0:
            atomic_write_snapshot(residue_to_mbss)

    return residue_to_mbss


async def load_proximal_drugs_dataset(species: str) -> pd.DataFrame:
    dataset_dir = config.directory.analysis / "drugs"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "proximal_drugs.csv"

    if dataset_path.exists():
        print("Proximal drug dataset already exists")
        return pd.read_csv(dataset_path)

    print("Building residue to MBS map")
    residue_mbs_map = get_heteroatom_residues(species)

    # Query PDB for drugbank IDs for each heteroatom residue.
    hetero_list = sorted(residue_mbs_map.keys())
    queries = [
        f"https://data.rcsb.org/rest/v1/core/chemcomp/{res}" for res in hetero_list
    ]

    responses: list[dict[str, Any]] = await tqdm_asyncio.gather(
        *[api_request(query) for query in queries],
        desc="Querying PDB for drugbank IDs",
        total=len(queries),
    )

    records: list[dict[str, Any]] = []
    for residue, response in zip(hetero_list, responses):
        if response is None:
            continue

        drugbank_id = response.get("rcsb_chem_comp_container_identifiers", {}).get(
            "drugbank_id"
        )
        if drugbank_id is None:
            continue

        associated_mbss = sorted(residue_mbs_map[residue])
        records.append(
            {
                "residue": residue,
                "drugbank_id": drugbank_id,
                "associated_mbss": associated_mbss,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(dataset_path, index=False)

    return df


def build_known_drug_targets_dataset(uniprot_ids: set[str]) -> pd.DataFrame:
    """Build a dataset of known drug targets from Drugbank for the given UniProt IDs."""
    path = config.directory.analysis / "drugs/known_drug_targets.csv"
    df = pd.read_csv(path)
    df = df[df["UniProt ID"].isin(uniprot_ids)]
    return df


def add_proximal_drug_annotations(
    network: nx.Graph, proximal_drugs_df: pd.DataFrame
) -> None:
    """Add drugbank IDs as set 'proximal_drugs' to nodes in the network.

    These represent drugs that have been shown experimentally to bind near the MBS."""
    for _, row in tqdm(
        proximal_drugs_df.iterrows(),
        total=len(proximal_drugs_df),
        desc="Adding drugbank IDs of drugs proximal to MBSs",
    ):
        drugbank_id = row["drugbank_id"]
        mbs_ids = ast.literal_eval(row["associated_mbss"])
        representative_ids = map(str, get_representative_ids(mbs_ids))

        for node in representative_ids:
            if not network.has_node(node):
                raise ValueError(f"Node {node} not found in the network")

            network.nodes[node]["proximal_drugs"].add(drugbank_id)


def add_known_drug_target_annotations(
    network: nx.Graph, known_drug_targets_df: pd.DataFrame
) -> None:
    """Add known drug target annotations to nodes in the network."""
    for _, row in tqdm(
        known_drug_targets_df.iterrows(),
        total=len(known_drug_targets_df),
        desc="Annotating known drug targets",
    ):
        uniprot_id = row["UniProt ID"]
        drug_ids = row["Drug IDs"].split("; ")

        for drug_id in drug_ids:
            for node in network.nodes:
                if network.nodes[node]["uniprot"] == uniprot_id:
                    network.nodes[node]["known_drugs"].add(drug_id)


def add_drug_attributes(network: nx.Graph) -> None:
    for n in network.nodes:
        for attr_name in ["proximal_drugs", "known_drugs"]:
            if attr_name not in network.nodes[n]:
                network.nodes[n][attr_name] = set()


class DrugNetwork(nx.Graph):
    """A NetworkX graph with drug-related attributes."""

    species: str

    @classmethod
    def from_existing_graph(cls, graph: nx.Graph, species: str) -> DrugNetwork:
        drug_network = cls()
        drug_network.add_nodes_from(graph.nodes(data=True))
        drug_network.add_edges_from(graph.edges(data=True))
        drug_network.species = species
        return drug_network

    @classmethod
    def from_network(cls, network: nx.Graph, species: str) -> DrugNetwork:
        """Build a drug network of the MBS network.

        Each node has attributes 'proximal_drugs' and 'known_drugs' which are sets of drugbank IDs.

        - 'proximal_drugs' are drugs that have been shown experimentally to bind near
          the metal binding site.
        - 'known_drugs' are drugs that according to Drugbank have been shown
          to target the MBS's encoding protein.

        Non-human nodes without any drug associations are removed from the network.
        """
        drug_network = cls.from_existing_graph(network, species)

        add_drug_attributes(drug_network)

        proximal_drugs_df = asyncio.run(load_proximal_drugs_dataset(species))

        uniprot_ids = {node[1]["uniprot"] for node in drug_network.nodes(data=True)}
        known_drug_targets_df = build_known_drug_targets_dataset(uniprot_ids)

        add_proximal_drug_annotations(drug_network, proximal_drugs_df)
        add_known_drug_target_annotations(drug_network, known_drug_targets_df)

        return drug_network


def prepare_enrichment_snapshot(
    network: DrugNetwork,
) -> tuple[list[tuple[int, int]], list[frozenset], list[frozenset]]:
    """Prepare a snapshot of the graph for enrichment analysis."""
    nodes = list(network.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    edges: list[tuple[int, int]] = []
    for u, v in network.edges():
        u_attrs = network.nodes[u]
        v_attrs = network.nodes[v]

        # Ignore edges between MBSs of the same protein.
        if u_attrs.get("uniprot") and u_attrs["uniprot"] == v_attrs.get("uniprot"):
            continue
        edges.append((idx[u], idx[v]))

    # Make immutable, pickle-friendly per-node sets
    proximal = [frozenset(network.nodes[n]["proximal_drugs"]) for n in nodes]
    known = [frozenset(network.nodes[n]["known_drugs"]) for n in nodes]
    return edges, proximal, known


def count_by_drug_on_snapshot(
    edges: list[tuple[int, int]], known: list[frozenset]
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for u, v in edges:
        hits = known[u].intersection(known[v])
        for d in hits:
            counts[d] += 1
    return counts


def permute_and_count(
    payload: tuple[list[tuple[int, int]], list[frozenset], list[frozenset]], seed: int
) -> dict[str, int]:
    edges, proximal, known = payload
    perm = list(range(len(known)))
    rnd = random.Random(int(seed))
    rnd.shuffle(perm)
    known_perm = [known[i] for i in perm]
    return count_by_drug_on_snapshot(edges, known_perm)


def drug_enrichment(network: DrugNetwork, shuffles: int) -> pd.DataFrame:
    """Evaluate enrichment of drug associations in the MBS network.

    Statistical ignificance is assessed against a null model generated
    from randomized network shuffles of the known drug target annotations
    (i.e. interactor drugs)."""

    # Prepare a fixed snapshot of the network for efficient permutation testing.
    payload = prepare_enrichment_snapshot(network)
    edges, _, known = payload
    observed_counts = count_by_drug_on_snapshot(edges, known)

    # Seeds for workers.
    rng = np.random.default_rng()
    seeds = rng.integers(0, 2**32 - 1, size=shuffles, dtype=np.uint32).tolist()

    # Run permutations in parallel.
    with tqdm_joblib(tqdm(desc="Permuting", total=shuffles)) as _:
        results = Parallel(n_jobs=-2)(
            delayed(permute_and_count)(payload, s) for s in seeds
        )

    null_counts: dict[str, list[int]] = defaultdict(list)
    for res in results:
        for drug, count in res.items():
            null_counts[drug].append(count)

    # Calculate enrichment statistics.
    volcano_data: list[dict[str, Any]] = []
    for drug, obs_count in observed_counts.items():
        null_dist = null_counts[drug]
        if np.sum(null_dist) == 0 and obs_count == 0:
            continue

        mean_null = np.mean(null_dist)
        std_null = np.std(null_dist)
        p_empirical = (np.sum(np.array(null_dist) >= obs_count) + 1) / (
            len(null_dist) + 1
        )
        enrichment_ratio = obs_count / mean_null if mean_null > 0 else np.inf
        z_score = (obs_count - mean_null) / std_null if std_null > 0 else np.inf

        volcano_data.append(
            {
                "drug": drug,
                "observed": obs_count,
                "mean_null": mean_null,
                "p_value": p_empirical,
                "log2_enrichment": np.log2(max(enrichment_ratio, 1e-6)),
                "z_score": z_score,
            }
        )

    # FDR correction by Benjamini-Hochberg.
    p_values = [data["p_value"] for data in volcano_data]
    p_adjusted = false_discovery_control(p_values, method="bh")
    for p_a, data in zip(p_adjusted, volcano_data):
        data["p_adjusted"] = p_a
        data["log10_p"] = -np.log10(p_a)

    volcano_df = pd.DataFrame(volcano_data)
    volcano_path = config.directory.analysis / "drugs/volcano_data.csv"
    volcano_df.to_csv(volcano_path, index=False)

    return volcano_df


class OffTargetRecord(BaseModel):
    drug: str = Field(..., description="DrugBank ID of the drug")
    drug_name: str = Field(..., description="Name of the drug")
    off_target_uniprot: str | None = Field(
        ..., description="UniProt ID of the candidate off-target"
    )
    off_target_name: str = Field(..., description="Name of the candidate off-target")
    proximal_drug_node_uniprot: str | None = Field(
        ..., description="UniProt ID of the neighbor with proximal drug evidence"
    )
    proximal_drug_node_name: str = Field(
        ..., description="Name of the neighbor with proximal drug evidence"
    )
    rmsd: float = Field(
        ..., description="RMSD between the off-target and proximal drug node"
    )
    interactor_drug_node_uniprot: str | None = Field(
        ..., description="UniProt ID of the node with interactor drug evidence"
    )
    interactor_drug_node_name: str | None = Field(
        ..., description="Name of the node with interactor drug evidence"
    )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OffTargetRecord):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.model_dump().items())))


def predict_drug_off_targets(
    network: DrugNetwork, target_drugs: set[str], enrichment_by_drug: dict[str, float]
) -> None:
    """Identify putative off-targets in the drug network."""
    all_drugbank_drugs = pd.read_csv(
        config.directory.analysis / "drugs/all_drugbank_drugs.csv"
    )
    records: set[OffTargetRecord] = set()
    for node in tqdm(
        network.nodes, total=network.number_of_nodes(), desc="Finding off-targets"
    ):
        node_attrs = network.nodes[node]
        neighbors = list(network.neighbors(node))
        for drug in node_attrs["known_drugs"]:
            if drug not in target_drugs:
                continue

            for cand_off_target in neighbors:
                add_off_target = False
                cand_off_target_attrs = network.nodes[cand_off_target]

                # Skip if non-human.
                if cand_off_target_attrs.get("species", "").lower() != "homo sapiens":
                    continue

                # Skip MBSs of the same protein.
                if node_attrs.get("uniprot") and node_attrs[
                    "uniprot"
                ] == cand_off_target_attrs.get("uniprot"):
                    continue

                # Skip if already known target.
                if drug in cand_off_target_attrs["known_drugs"]:
                    continue

                edge_data = network.get_edge_data(node, cand_off_target)

                # Node both a known drug target and the drug binds near its MBS.
                if drug in node_attrs["proximal_drugs"]:
                    add_off_target = True
                    proximal_drug_node_uniprot = node_attrs["uniprot"]
                    proximal_drug_node_name = node_attrs["peptide"]
                else:
                    for neighbor in neighbors:
                        if neighbor == cand_off_target:
                            continue

                        neighbor_attrs = network.nodes[neighbor]
                        # Node is a known target and one of its neighbors has proximal drug evidence.
                        if drug in neighbor_attrs["proximal_drugs"]:
                            add_off_target = True
                            proximal_drug_node_uniprot = neighbor_attrs["uniprot"]
                            proximal_drug_node_name = neighbor_attrs["peptide"]
                            break

                if add_off_target:
                    drug_name = all_drugbank_drugs.loc[
                        all_drugbank_drugs["DrugBank ID"] == drug, "Common name"
                    ].values[0]
                    records.add(
                        OffTargetRecord(  # type: ignore[call-arg]
                            drug=drug,
                            drug_name=drug_name,
                            off_target_uniprot=cand_off_target_attrs["uniprot"],
                            off_target_name=cand_off_target_attrs["peptide"],
                            proximal_drug_node_uniprot=proximal_drug_node_uniprot,
                            proximal_drug_node_name=proximal_drug_node_name,
                            rmsd=edge_data["rmsd"],
                            interactor_drug_node_uniprot=node_attrs["uniprot"],
                            interactor_drug_node_name=node_attrs["peptide"],
                        )
                    )

    # Print unique drug-off-target combinations.
    unique_combinations = set()
    for rec in records:
        unique_combinations.add((rec.drug, rec.off_target_uniprot))
    print(f"Unique drug-off-target combinations: {len(unique_combinations)}")

    # Print number of unique drugs.
    unique_drugs = set()
    for rec in records:
        unique_drugs.add(rec.drug)
    print(f"Unique drugs with off-targets: {len(unique_drugs)}")

    # Print number of unique proteins (i.e. UniProt IDs).
    unique_proteins = set()
    for rec in records:
        unique_proteins.add(rec.off_target_uniprot)
    print(f"Unique proteins with off-targets: {len(unique_proteins)}")

    combined_records: dict[tuple[str, str], OffTargetRecord] = {}
    for rec in records:
        key = (rec.drug, rec.off_target_uniprot)
        if key in combined_records:
            existing_rec = combined_records[key]
            if rec.rmsd < existing_rec.rmsd:
                combined_records[key] = rec
        else:
            combined_records[key] = rec

    sorted_records = sorted(list(combined_records.values()), key=lambda x: x.rmsd)
    df = pd.DataFrame([rec.model_dump() for rec in sorted_records])
    df["log2_enrichment"] = df["drug"].map(enrichment_by_drug)
    df.to_csv(
        config.directory.analysis / "drugs/predicted_off_targets.csv",
        index=False,
    )


def main() -> None:
    drug_network_path = config.directory.analysis / "drugs/attributes.json"
    if not drug_network_path.exists():
        drug_network_path.parent.mkdir(parents=True, exist_ok=True)
        network = read_network_json("MBSNetwork")
        drug_network = DrugNetwork.from_network(network, species="all")
        write_network_json(drug_network, config.directory.analysis / "drugs")
    else:
        network = read_network_json(name="", root=config.directory.analysis / "drugs")
        drug_network = DrugNetwork.from_existing_graph(network, species="all")

    all_known_drugs = set()
    all_proximal_drugs = set()
    for n in drug_network.nodes:
        all_known_drugs.update(drug_network.nodes[n]["known_drugs"])
        all_proximal_drugs.update(drug_network.nodes[n]["proximal_drugs"])
    print(f"Drugs with known MBS protein interactions: {len(all_known_drugs)}")
    print(f"Proximal drugs: {len(all_proximal_drugs)}")

    nodes_with_known_drug = sum(
        1 for n in drug_network.nodes if len(drug_network.nodes[n]["known_drugs"]) > 0
    )
    nodes_with_proximal = sum(
        1
        for n in drug_network.nodes
        if len(drug_network.nodes[n]["proximal_drugs"]) > 0
    )
    print(f"MBS nodes of known drug target proteins: {nodes_with_known_drug}")
    print(f"MBS nodes annotated with proximal drugs: {nodes_with_proximal}")

    drug_to_uniprot_nodes = defaultdict(lambda: defaultdict(set))  # type: ignore[var-annotated]
    for n, attrs in drug_network.nodes(data=True):
        uniprot = attrs.get("uniprot")
        for d in attrs.get("known_drugs", []):
            drug_to_uniprot_nodes[d][uniprot].add(n)

    drug_counts_dedup = {
        d: len(uniprot_map) for d, uniprot_map in drug_to_uniprot_nodes.items()
    }
    multi_known_drugs = {d for d, c in drug_counts_dedup.items() if c >= 2}

    print(
        f"Known drugs annotated to >=2 MBS-encoding proteins (interactor drugs): {len(multi_known_drugs)}"
    )

    # Run enrichment analysis.
    volcano_data_path = config.directory.analysis / "drugs/volcano_data.csv"
    #if not volcano_data_path.exists():
    df_volcano = drug_enrichment(drug_network, shuffles=10000)
    ##else:
    #    df_volcano = pd.read_csv(volcano_data_path)

    # Find significant over enrichments (adjusted p-value).
    sig_over = df_volcano[
        (df_volcano["p_adjusted"] < 0.05) & (df_volcano["log2_enrichment"] > 0)
    ]
    non_sig = df_volcano[df_volcano["p_adjusted"] >= 0.05]
    print(
        f"Significant over enrichments: {len(sig_over)} vs. Non-significant: {len(non_sig)}"
    )
    enrichment_by_drug = sig_over.set_index("drug")["log2_enrichment"].to_dict()

    # Interactor drugs with significant over enrichment that also show
    # proximal binding evidence.
    sig_proximal = set(sig_over["drug"]).intersection(all_proximal_drugs)
    print(
        f"Significantly over enriched drugs with proximal binding evidence: {len(sig_proximal)}"
    )

    # Identify putative off-targets.
    predict_drug_off_targets(
        drug_network,
        target_drugs=set(sig_over["drug"].tolist()),
        enrichment_by_drug=enrichment_by_drug,
    )


if __name__ == "__main__":
    main()
