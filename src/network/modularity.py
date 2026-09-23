from __future__ import annotations

from typing import TYPE_CHECKING

import igraph as ig  # type: ignore[import-untyped]
import leidenalg  # type: ignore[import-untyped]
import networkx as nx
import py4cytoscape as py4c  # type: ignore[import-untyped]
from database import Session
from database.datamodel.models import MetalBindingSite
from preprocessing.structure import build_structure
from sqlalchemy import select

from network.network import read_network_json

if TYPE_CHECKING:
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure


def calculate_leiden_modularity(graph: nx.Graph) -> list[frozenset]:
    """Calculate the Leiden modularity of a network."""
    igraph_graph = ig.Graph.from_networkx(graph)

    # Compute community partition using Leiden algorithm
    partition = leidenalg.find_partition(
        igraph_graph, leidenalg.ModularityVertexPartition, n_iterations=-1, seed=1
    )

    return [
        frozenset(igraph_graph.vs[vertex]["_nx_name"] for vertex in cluster)
        for cluster in partition
    ]


def get_dominant_motif(structure: Structure, coord: tuple[float, float, float]) -> str:
    """Get the dominant 4-residue motif surrounding a metal coordinate.

    Returns an unordered residue multiset as a hyphen-joined, alphabetically
    sorted 3-letter codes (e.g., "CYS-CYS-HIS-HIS").
    """

    cx, cy, cz = coord

    # Track minimum heavy-atom distance per residue (across the entire structure).
    best: dict[tuple[str, int, str], tuple[float, Residue]] = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip non-standard residues and waters
                hetflag = residue.id[0]
                if hetflag != " ":
                    continue

                # Ensure residue has a resname
                resname = getattr(residue, "resname", None)
                if not resname:
                    continue

                # Compute min heavy-atom distance for this residue
                min_d2: float | None = None
                for atom in residue.get_atoms():
                    # Skip hydrogens if present
                    element = getattr(atom, "element", "")
                    if element == "H":
                        continue
                    # Fallback: some PDBs may not set element; skip by name heuristic
                    if not element and atom.get_id().startswith("H"):
                        continue

                    x, y, z = atom.get_coord()
                    d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz)
                    if (min_d2 is None) or (d2 < min_d2):
                        min_d2 = d2

                if min_d2 is None:
                    continue

                key = (chain.id, residue.id[1], residue.id[2])  # chain, resseq, icode
                prev = best.get(key)
                if (prev is None) or (min_d2 < prev[0]):
                    best[key] = (min_d2, residue)

    # Take the 4 closest residues by min heavy-atom distance.
    closest = sorted(best.values(), key=lambda t: t[0])[:4]
    resnames = sorted(res.resname for _, res in closest)
    return "-".join(resnames)


def compute_dominant_module_metals(graph: nx.Graph, modules: list[frozenset]) -> None:
    """Compute the dominant metal type for each module."""
    module_metals: dict[int, str] = {}

    for module_id, module_nodes in enumerate(modules, start=1):
        metal_counts: dict[str, int] = {}

        for node in module_nodes:
            node_attr = graph.nodes[node]
            metal = node_attr["ligand"]
            metal_counts[metal] = metal_counts.get(metal, 0) + 1

        dominant_metal = max(metal_counts, key=metal_counts.get)
        module_metals[module_id] = dominant_metal
        print(
            f"Module {module_id}: Dominant metal = {dominant_metal} (freq={metal_counts[dominant_metal] / len(module_nodes):.2%})"
        )


def compute_dominant_module_motifs(graph: nx.Graph, modules: list[frozenset]) -> None:
    """Compute the dominant 4-residue motif surrounding the MBS metal coordinate."""
    module_motifs: dict[int, str] = {}

    for module_id, module_nodes in enumerate(modules, start=1):
        motif_counts: dict[str, int] = {}

        with Session() as session:
            for node in module_nodes:
                node_attr = graph.nodes[node]
                mbs_id = node_attr["id"]
                mbs = session.execute(
                    select(MetalBindingSite).where(MetalBindingSite.id == mbs_id)
                ).scalar_one()
                structure = build_structure(mbs.assembly)
                motif = get_dominant_motif(structure, mbs.ligand_coord)
                motif_counts[motif] = motif_counts.get(motif, 0) + 1

        dominant_motif = max(motif_counts, key=motif_counts.get)
        module_motifs[module_id] = dominant_motif
        print(
            f"Module {module_id}: Dominant motif = {dominant_motif} (freq={motif_counts[dominant_motif] / len(module_nodes):.2%})"
        )


if __name__ == "__main__":
    network = read_network_json("MBSNetwork")

    # Get largest connected component.
    largest_cc = max(nx.connected_components(network), key=len)
    largest_subgraph = network.subgraph(largest_cc)
    print(
        f"Largest subnetwork of size: {largest_subgraph.number_of_nodes()} nodes and {largest_subgraph.number_of_edges()} edges"
    )

    # Calculate Leiden modularity.
    modules = calculate_leiden_modularity(largest_subgraph)

    # Select 8 largest modules for further analysis.
    modules = sorted(modules, key=len, reverse=True)[:8]

    compute_dominant_module_metals(largest_subgraph, modules)
    compute_dominant_module_motifs(largest_subgraph, modules)

    # --- Add Cytoscape-friendly module labels A–H for modules 1–8 ---
    module_letters = "ABCDEFGH"
    max_labeled = min(10, len(modules))

    # Default for nodes not in modules 1–8 (helps Cytoscape filtering)
    nx.set_node_attributes(largest_subgraph, "Other", "module_label")
    nx.set_node_attributes(largest_subgraph, 0, "module_id")

    for module_id in range(1, max_labeled + 1):
        letter = module_letters[module_id - 1]
        for node in modules[module_id - 1]:
            largest_subgraph.nodes[node]["module_id"] = module_id
            largest_subgraph.nodes[node]["module_label"] = letter

    # Add normalized RMSD attribute for visualization (edge attribute).
    for edge in largest_subgraph.edges:
        rmsd = largest_subgraph.edges[edge]["rmsd"]
        max_rmsd = 1
        rmsd_norm = 1 - (rmsd / max_rmsd)
        largest_subgraph.edges[edge]["rmsd_norm"] = rmsd_norm

    # Visualize the largest subgraph with Py4Cytoscape.
    py4c.create_network_from_networkx(
        largest_subgraph, title="Component_alpha", collection="MBSNetworks"
    )
    py4c.set_node_size_default(45)
    py4c.set_node_shape_default("ellipse")
