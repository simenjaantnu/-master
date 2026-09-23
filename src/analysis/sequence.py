from __future__ import annotations

import ast
import itertools
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import igraph  # type: ignore[import-untyped]
import networkx as nx
import numpy as np
import requests
from Bio import SeqIO
from Bio.Align import Alignment, PairwiseAligner
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from config import config
from database import Session
from database.datamodel.models import MetalBindingSite
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from network.network import read_network_json
from network.utils import tqdm_joblib  # type: ignore[import-untyped]
from sqlalchemy import select
from tqdm import tqdm


def run_ssearch36(
    query_fa: Path,
    subject_fa: Path,
    ssearch_bin: Path = Path("/MBSNetwork/fasta-36.1.1/bin/ssearch36"),
    shuffles_k: int = 1000,
    effective_db_Z: int = 20000,
    matrix: str = "BP62",
) -> Path:
    """Run ssearch36 non-interactively and return the path to a temporary output file."""
    tmp = tempfile.NamedTemporaryFile(prefix="ssearch36_", suffix=".txt", delete=False)
    out_path = tmp.name
    tmp.close()

    cmd = [
        str(ssearch_bin),
        "-q",
        "-k",
        str(shuffles_k),
        "-Z",
        str(effective_db_Z),
        "-s",
        matrix,
        "-O",
        out_path,
        str(query_fa),
        str(subject_fa),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            "ssearch36 failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )

    return Path(out_path)


_EVALUE_LINE_RE = re.compile(
    r"""^
        \S+
        .*?
        \s+\(?\s*\d+\s*\)?
        \s+(\d+)
        \s+([0-9]+(?:\.[0-9]+)?)
        \s+([0-9.eE+-]+)
        \s*$
    """,
    re.VERBOSE,
)

_NO_HITS_RE = re.compile(r"^!!\s+No sequences with E\(\)\s*<\s*10\s*$")


def parse_evalue_from_ssearch_report(
    report_path: Path,
    effective_db_Z: int | None = None,
    no_hit_evalue: float = 10.0,
) -> float:
    """Parse E-value from an ssearch36 human-readable report.

    If effective_db_Z is provided, we additionally sanity-check that the header line
    contains E(<Z>) to avoid silently mixing mismatched Z values.
    """
    text = report_path.read_text(errors="replace").splitlines()

    # Early exit: explicit "no hits" line
    if any(_NO_HITS_RE.match(line.strip()) for line in text):
        return no_hit_evalue

    # Optional sanity check on the header line: "The best scores are: ... E(20000)"
    if effective_db_Z is not None:
        expected = f"E({effective_db_Z})"
        if not any(expected in line for line in text):
            raise ValueError(
                f"Expected header to contain '{expected}' but did not find it in {report_path}"
            )

    # Parse first hit line after "The best scores are:"
    in_scores = False
    for line in text:
        if line.startswith("The best scores are:"):
            in_scores = True
            continue
        if not in_scores:
            continue

        if not line.strip():
            continue

        m = _EVALUE_LINE_RE.match(line.rstrip("\n"))
        if m:
            e_str = m.group(3)
            try:
                return float(e_str)
            except ValueError as e:
                raise ValueError(
                    f"Failed to parse E-value '{e_str}' from line: {line}"
                ) from e

        # If we hit the explicit "no hits" line after entering the block
        if _NO_HITS_RE.match(line.strip()):
            return no_hit_evalue

    raise ValueError(f"Could not find a parsable best-hit E-value in {report_path}")


def align_sequences(
    seq_1: str, seq_2: str, mode: str = "global", scoring: str = "blastp"
) -> float:
    def percentage_identity(alignment: Alignment) -> float:
        i = 0
        for a in range(len(alignment[0])):
            column = alignment[:, a]
            if column == len(column) * column[0]:
                i += 1
        return 100 * i / len(alignment[0])

    aligner = PairwiseAligner(scoring=scoring)
    aligner.mode = mode
    alignments = aligner.align(seq_1, seq_2)

    return percentage_identity(alignments[0])


def compute_seq_identity(seq_a: str, seq_b: str) -> dict[str, float]:
    try:
        identity = align_sequences(seq_a, seq_b)
    except Exception:
        return {"identity": 0.0}
    return {"identity": identity}


def _chunked(iterable, n):
    it = iter(iterable)
    while True:
        block = list(itertools.islice(it, n))
        if not block:
            break
        yield block


def _parse_fasta_to_dict(text: str) -> dict[str, str]:
    seqs, acc, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if acc is not None:
                seqs[acc] = "".join(buf)
            parts = line[1:].split("|")
            acc = parts[1] if len(parts) > 1 else parts[0].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if acc is not None:
        seqs[acc] = "".join(buf)
    return seqs


def fetch_and_update_sequences(network: nx.Graph) -> None:
    path = config.directory.networks / f"{network.name}/uniprot_sequences.fasta"
    if path.exists():
        seq_map = _parse_fasta_to_dict(path.read_text())
        for node, data in network.nodes(data=True):
            acc = data.get("uniprot")
            if acc and acc in seq_map:
                data["sequence"] = seq_map[acc]
        return

    acc_to_nodes = defaultdict(list)
    for node, data in network.nodes(data=True):
        acc = data.get("uniprot")
        if acc:
            acc_to_nodes[acc].append(node)

    ids = sorted(acc_to_nodes.keys())
    if not ids:
        return

    headers = {
        "User-Agent": "AlmaasLab/1.0 (vetle.simensen@ntnu.no)",
        "Accept": "text/x-fasta",
    }
    base_url = "https://rest.uniprot.org/uniprotkb/stream"
    chunk_size = 100

    seq_map: dict[str, str] = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out, tqdm(
        total=len(ids), desc="Fetching UniProt sequences"
    ) as pbar:
        for block in _chunked(ids, chunk_size):
            query = "(" + " OR ".join(f"accession:{acc}" for acc in block) + ")"
            params = {"format": "fasta", "compressed": "false", "query": query}
            attempts, backoff = 0, 1.0
            while True:
                try:
                    r = requests.get(
                        base_url, params=params, headers=headers, timeout=90
                    )
                    if r.status_code == 200:
                        text = r.text
                        out.write(text)
                        seqs = _parse_fasta_to_dict(text)
                        seq_map.update(seqs)
                        pbar.update(len(block))
                        break
                    if r.status_code == 429:
                        ra = r.headers.get("Retry-After")
                        sleep_s = float(ra) if ra and ra.isdigit() else backoff
                        time.sleep(sleep_s)
                    elif 500 <= r.status_code < 600:
                        time.sleep(backoff)
                    else:
                        if r.text:
                            out.write(r.text)
                            seq_map.update(_parse_fasta_to_dict(r.text))
                        pbar.update(len(block))
                        break
                except requests.RequestException:
                    time.sleep(backoff)
                attempts += 1
                backoff = min(backoff * 2, 30.0)
                if attempts >= 5:
                    pbar.update(len(block))
                    break

    for acc, seq in seq_map.items():
        for node in acc_to_nodes[acc]:
            network.nodes[node]["sequence"] = seq


def add_sequence_identity(graph: nx.Graph) -> None:
    with tqdm_joblib(
        tqdm(desc="Computing sequence identity", total=len(graph.edges))
    ) as _:
        results = Parallel(n_jobs=-2)(
            delayed(compute_seq_identity)(
                graph.nodes[u].get("sequence", ""), graph.nodes[v].get("sequence", "")
            )
            for u, v in graph.edges
        )
    attr = dict(zip(graph.edges, results))
    nx.set_edge_attributes(graph, attr)


def get_edge_dict(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    edge_dict: dict[tuple[int, int], dict[str, float]] = {}
    with path.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            node1, node2 = int(parts[0]), int(parts[1])
            attr = ast.literal_eval(parts[2])
            if attr["identity"] == 0 or attr["identity"] == 100:
                continue
            edge_dict[(node1, node2)] = attr

    return edge_dict


def identify_geometry_conserved_sequence_divergent_pairs(
    edge_dict: dict[tuple[int, int], dict[str, float]],
    output_path: Path,
    rmsd_threshold: float = 0.8,
    identity_threshold: float = 25.0,
    normalized_rmsd_threshold: float = 0.375,
) -> None:
    selected_keys: list[tuple[tuple[int, int], float, float]] = []
    for key, attr in edge_dict.items():
        x_val = 1 - (attr["rmsd"] / rmsd_threshold)
        y_val = attr["identity"]
        if x_val > normalized_rmsd_threshold and y_val < identity_threshold:
            selected_keys.append((key, attr["identity"], attr["rmsd"]))

    selected_keys.sort(key=lambda x: (x[2]))
    results = np.empty((len(selected_keys), 2), dtype=float)
    j = 0

    with Session() as session:
        with output_path.open("w") as f:
            f.write(
                "MBS id 1\tLigand\tSpecies\tPDB entry\tName\tUniprot\tEC\tMBS id 2\tLigand\tSpecies\tPDB entry\tName\tUniprot\tEC\tGlobal seq identity\tE value (local seq alignment)\tRMSD\n"
            )
            protein_pairs: set[tuple[str, str]] = set()
            ec_overlap_count = 0
            for i, (key, identity, rmsd) in enumerate(
                tqdm(
                    selected_keys,
                    desc="Finding geometry-conserved sequence-divergent pairs",
                )
            ):
                mbs_1 = session.execute(
                    select(MetalBindingSite).where(MetalBindingSite.id == key[0])
                ).scalar_one()
                mbs_2 = session.execute(
                    select(MetalBindingSite).where(MetalBindingSite.id == key[1])
                ).scalar_one()
                ligand_1 = (
                    mbs_1.ligand_form.ligands[0].pdb_id
                    if mbs_1.ligand_form.ligands
                    else None
                )
                ligand_2 = (
                    mbs_2.ligand_form.ligands[0].pdb_id
                    if mbs_2.ligand_form.ligands
                    else None
                )

                # Skip if same UniProt accession
                uniprot_1 = mbs_1.peptide.uniprot
                uniprot_2 = mbs_2.peptide.uniprot
                if uniprot_1 is not None and uniprot_1 == uniprot_2:
                    continue

                # Skip if UniProt of peptides in the PDB entry overlap.
                pdb_1_uniprots = {
                    p.uniprot
                    for p in mbs_1.peptide.entry.peptides
                    if p.uniprot is not None
                }
                pdb_2_uniprots = {
                    p.uniprot
                    for p in mbs_2.peptide.entry.peptides
                    if p.uniprot is not None
                }
                if pdb_1_uniprots.intersection(pdb_2_uniprots):
                    continue

                # Skip if global sequence alignment of all peptides in the PDB entry
                # do not reach threshold.
                seq_1s = [
                    p.sequence for p in mbs_1.peptide.entry.peptides if p.sequence
                ]
                seq_2s = [
                    p.sequence for p in mbs_2.peptide.entry.peptides if p.sequence
                ]
                if not seq_1s or not seq_2s:
                    continue

                max_identity = 0.0
                for seq_1 in seq_1s:
                    for seq_2 in seq_2s:
                        identity = align_sequences(seq_1, seq_2)
                        if identity > max_identity:
                            max_identity = identity

                if max_identity >= identity_threshold:
                    continue

                identity = align_sequences(
                    mbs_1.peptide.sequence, mbs_2.peptide.sequence
                )

                # Record protein pairs to account for duplicates.
                if uniprot_1 and uniprot_2:
                    pair_key = tuple(sorted([uniprot_1, uniprot_2]))
                    if pair_key not in protein_pairs:
                        # Record if EC numbers overlap.
                        ec_1 = mbs_1.peptide.ec
                        ec_2 = mbs_2.peptide.ec
                        if ec_1 and ec_2:
                            if set(ec_1.split(",")).intersection(set(ec_2.split(","))):
                                ec_overlap_count += 1

                    protein_pairs.add(pair_key)

                # Skip if showing significant local sequence similarity (E-value < 0.001).
                e_value = compute_local_seq_similarity(
                    mbs_1.peptide.sequence, mbs_2.peptide.sequence
                )
                results[j, :] = [1 - (rmsd / 0.8), e_value]
                j += 1

                if e_value < 0.001:
                    continue

                f.write(
                    f"{mbs_1.id}\t{ligand_1}\t{mbs_1.peptide.species}\t{mbs_1.peptide.entry_pdb_id}\t{mbs_1.peptide.name}\t{uniprot_1}\t{mbs_1.peptide.ec}"
                    f"\t{mbs_2.id}\t{ligand_2}\t{mbs_2.peptide.species}\t{mbs_2.peptide.entry_pdb_id}\t{mbs_2.peptide.name}\t{uniprot_2}\t{mbs_2.peptide.ec}\t"
                    f"{identity}\t{e_value}\t{rmsd}\n"
                )

    # Save results array for downstream analysis.
    results = results[:j]
    np.save(output_path.with_suffix(".npy"), results)

    print(f"Found {len(protein_pairs)} unique protein pairs.")
    print(f"Of these, {ec_overlap_count} pairs have overlapping EC numbers.")


def geometry_conserved_sequence_divergent_pairs_statistics(path: Path) -> None:
    total_lines = 0
    unique_proteins: set[str] = set()

    ec3_overlap_count = 0

    same_ligand_pair_rows = 0
    different_ligand_pair_rows = 0

    ligand_counts = Counter()

    def clean_ec_tokens(ec_field: str) -> set[str]:
        bad = {"none", "na", "nan", "null", "-", ""}
        out = set()
        for e in ec_field.split(","):
            t = e.strip()
            if t.lower() in bad:
                continue
            out.add(t)
        return out

    def ec_to_ec3(ec: str) -> str | None:
        """Map full EC (a.b.c.d) to third level (a.b.c). Returns None if malformed."""
        parts = ec.split(".")
        if len(parts) < 3:
            return None
        a, b, c = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not (a and b and c):
            return None
        return f"{a}.{b}.{c}"

    with path.open() as f:
        next(f)  # skip header
        for line in f:
            total_lines += 1
            parts = line.rstrip("\n").split("\t")

            lig1 = parts[1].strip()
            lig2 = parts[8].strip()

            uni1 = parts[5].strip()
            uni2 = parts[12].strip()

            ec1 = parts[6].strip()
            ec2 = parts[13].strip()

            if uni1:
                unique_proteins.add(uni1)
            if uni2:
                unique_proteins.add(uni2)

            # EC overlap (row-wise, full EC strings)
            if ec1 and ec2:
                ec1_set = clean_ec_tokens(ec1)
                ec2_set = clean_ec_tokens(ec2)

                # EC overlap at 3rd level (a.b.c)
                ec1_3 = {ec_to_ec3(e) for e in ec1_set}
                ec2_3 = {ec_to_ec3(e) for e in ec2_set}
                ec1_3.discard(None)
                ec2_3.discard(None)
                if ec1_3 and ec2_3 and (ec1_3 & ec2_3):
                    ec3_overlap_count += 1

            # Ligand pair comparison (row-wise)
            if lig1 == lig2:
                same_ligand_pair_rows += 1
            else:
                different_ligand_pair_rows += 1

            # Overall ligand distribution (across both ligand columns)
            if lig1:
                ligand_counts[lig1] += 1
            if lig2:
                ligand_counts[lig2] += 1

    print(f"Total MBS pairs (rows): {total_lines}")
    print(f"Unique UniProt proteins: {len(unique_proteins)}")
    print(f"EC overlap count (row-wise, EC third level a.b.c): {ec3_overlap_count}")

    print(f"Rows with same ligand pair (equal): {same_ligand_pair_rows}")
    print(f"Rows with different ligand pair (different): {different_ligand_pair_rows}")

    print("Overall ligand distribution (counts across both ligand columns):")
    for lig, cnt in ligand_counts.most_common():
        print(f"  {lig}\t{cnt}")


def compute_local_seq_similarity(seq_1: str, seq_2: str) -> float:
    """Compute local sequence similarity E-value using FASTA ssearch."""
    record_1 = SeqRecord(Seq(seq_1), id="query")
    record_2 = SeqRecord(Seq(seq_2), id="subject")

    # Make temp dir and fasta files for query and subject.
    with tempfile.TemporaryDirectory() as tmpdir:
        query_fa = Path(tmpdir) / "query.fa"
        subject_fa = Path(tmpdir) / "subject.fa"
        SeqIO.write(record_1, query_fa, "fasta")
        SeqIO.write(record_2, subject_fa, "fasta")

        # Run ssearch36
        report_path = run_ssearch36(
            query_fa,
            subject_fa,
            ssearch_bin=Path("/MBSNetwork/fasta-36.1.1/bin/ssearch36"),
            shuffles_k=1000,
            effective_db_Z=20000,
            matrix="BP62",
        )

        e_value = parse_evalue_from_ssearch_report(report_path, effective_db_Z=20000)

    return e_value


def permutation_test_induced_subgraph_avg_clustering(
    network: nx.Graph, output_path: Path, R: int = 1000, seed: int = 1
) -> None:
    # Read selected edges (node IDs are strings)
    edges: list[tuple[str, str]] = []
    with output_path.open() as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            edges.append((parts[0], parts[7]))

    m = len(edges)

    # Build igraph once, preserving NetworkX node names
    g = igraph.Graph.from_networkx(network)
    # Map node name -> vertex index
    name_to_vid = {v["_nx_name"]: v.index for v in g.vs}

    # Observed node-induced subgraph from selected edges
    obs_vids = set()
    for u, v in edges:
        obs_vids.add(name_to_vid[u])
        obs_vids.add(name_to_vid[v])

    obs_sub = g.induced_subgraph(list(obs_vids))
    obs_c = obs_sub.transitivity_avglocal_undirected(mode="zero")
    print(
        f"Observed subgraph has {obs_sub.vcount()} nodes and {obs_sub.ecount()} edges."
    )
    print(f"Observed average clustering: {obs_c:.6g}")

    E = g.ecount()
    rng = np.random.default_rng(seed)

    null_cs = np.empty(R, dtype=float)
    for r in tqdm(range(R), desc="Permutation test"):
        eids = rng.choice(E, size=m, replace=False)
        vids = set()
        for eid in eids:
            a, b = g.es[int(eid)].tuple
            vids.add(a)
            vids.add(b)
        sub = g.induced_subgraph(list(vids))
        null_cs[r] = sub.transitivity_avglocal_undirected(mode="zero")

    p = (1.0 + float(np.sum(null_cs >= obs_c))) / (R + 1.0)
    print(f"Null mean ± sd: {null_cs.mean():.6g} ± {null_cs.std(ddof=1):.6g} (R={R})")
    print(f"Empirical one-sided p-value (null >= obs): {p:.6g}")


def main() -> None:
    name = "MBSNetwork"
    edge_attr_path = (
        config.directory.networks / f"{name}/edge_attributes_with_seq_identity.tsv"
    )
    network = read_network_json(name=name)
    if not edge_attr_path.exists():
        fetch_and_update_sequences(network)
        add_sequence_identity(network)
        nx.write_edgelist(network, edge_attr_path, delimiter="\t")

    edge_dict = get_edge_dict(edge_attr_path)
    output_path = (
        config.directory.networks
        / f"{name}/geometry_conserved_sequence_divergent_pairs.tsv"
    )

    identify_geometry_conserved_sequence_divergent_pairs(
        edge_dict,
        output_path,
        rmsd_threshold=1,
        identity_threshold=25.0,
        normalized_rmsd_threshold=0.375,
    )
    geometry_conserved_sequence_divergent_pairs_statistics(output_path)
    permutation_test_induced_subgraph_avg_clustering(network, output_path)


if __name__ == "__main__":
    main()
