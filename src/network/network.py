from __future__ import annotations

import ast
import json
import logging
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from config import config
from database import Session
from preprocessing.dataset import Dataset
from tqdm import tqdm

from network.utils import get_site_attributes

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def build_network(edge_bunch: list[tuple[int, int, float]]) -> nx.Graph:
    network: nx.Graph = nx.Graph()
    network.add_weighted_edges_from(edge_bunch)
    return network


def _encode(obj: Any) -> Any:
    if isinstance(obj, set):
        return {"__type__": "set", "items": list(obj)}
    if isinstance(obj, list):
        return [_encode(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__type__") == "set":
            return set(_decode(x) for x in obj.get("items", []))
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(x) for x in obj]
    return obj


def write_network_json(graph: nx.Graph, path: Path) -> None:
    """Writes network to file with node attributes."""
    graph_path = path / "network.edgelist"
    node_attr_path = path / "attributes.json"

    nx.write_edgelist(graph, graph_path, delimiter="\t")

    node_attrs = {str(node): dict(attr) for node, attr in graph.nodes(data=True)}
    with node_attr_path.open("w", encoding="utf-8") as f:
        json.dump(_encode(node_attrs), f, ensure_ascii=False, indent=2)


def read_network_json(name: str, root: Path = config.directory.networks) -> nx.Graph:
    """Reads network from file with node attributes."""
    graph_path = root / name / "network.edgelist"
    node_attr_path = root / name / "attributes.json"

    with node_attr_path.open("r", encoding="utf-8") as f:
        node_attrs = _decode(json.load(f))

    # Initialize graph with all nodes and their attributes (including isolated nodes)
    graph: nx.Graph = nx.Graph(name=name)
    graph.add_nodes_from((str(node), attrs) for node, attrs in node_attrs.items())

    # Read edges; parse attrs safely
    with graph_path.open() as f:
        for line in f:
            parts = line.strip().split(None, 2)
            if len(parts) == 2:
                u, v = parts
                graph.add_edge(str(u), str(v))
            elif len(parts) == 3:
                u, v, attr_str = parts
                u, v = str(u), str(v)
                try:
                    attr = ast.literal_eval(attr_str)
                except (ValueError, SyntaxError):
                    attr = {}
                if isinstance(attr, dict) and "rmsd" in attr:
                    graph.add_edge(u, v, rmsd=attr["rmsd"])
                else:
                    graph.add_edge(u, v)

    # Convert Uniprot floats to None.
    for _, attrs in graph.nodes(data=True):
        if (
            "uniprot" in attrs
            and isinstance(attrs["uniprot"], float)
            and np.isnan(attrs["uniprot"])
        ):
            attrs["uniprot"] = None

    return graph


def create_network_from_alignments(
    alignment_files: list[Path],
    threshold: float,
    name: str = "MBSNetwork",
    remove_isolated: bool = False,
    to_alignment_file: Path | None = None,
) -> nx.Graph:
    """Construct a network from the provided alignment files.

    Optionally include additional alignments from TO-based realignment."""
    network: nx.Graph[int] = nx.Graph(name=name)

    with Session() as session:
        dataset = Dataset(config)
        dataset.get_representatives(session)
        mbs_attrs = {site.id: get_site_attributes(site) for site in dataset.mbss}

        logger.info("Creating network from alignments...")
        for network_path in alignment_files:
            alignments: np.ndarray = np.load(network_path)
            for alignment in tqdm(alignments):
                source_id: int = alignment[0].astype(int)
                target_id: int = alignment[1].astype(int)
                rmsd: float = alignment[2]

                # Add nodes and attributes
                if not network.has_node(source_id):
                    source_attrs = mbs_attrs[source_id]
                    network.add_node(source_id, **source_attrs)
                if not network.has_node(target_id):
                    target_attrs = mbs_attrs[target_id]
                    network.add_node(target_id, **target_attrs)

                if alignment[2] <= threshold:
                    network.add_edge(source_id, target_id, rmsd=rmsd)

        if to_alignment_file:
            logger.info("Adding edges from TO-based realignments...")
            alignments = np.load(to_alignment_file)
            for alignment in tqdm(alignments):
                source_id = alignment[0].astype(int)
                target_id = alignment[1].astype(int)
                rmsd = alignment[2]

                if rmsd <= threshold:
                    network.add_edge(source_id, target_id, rmsd=rmsd)

        if remove_isolated:
            network.remove_nodes_from(list(nx.isolates(network)))

    return network


if __name__ == "__main__":
    name = "MBSNetwork"
    network = create_network_from_alignments(
        alignment_files=list(config.directory.alignments.glob("*.npy")),
        threshold=1,
        name=name,
        remove_isolated=False,
        to_alignment_file=None,
    )
    print(f"Size of the network: {network.number_of_nodes()} nodes")
    print(f"Number of edges: {network.number_of_edges()} edges")
    network = create_network_from_alignments(
        alignment_files=list(config.directory.alignments.glob("*.npy")),
        threshold=1,
        name=name,
        remove_isolated=False,
        to_alignment_file=next(
            (config.directory.alignments / "to_realignment").glob("*.npy")
        ),
    )
    print(f"Size of the network: {network.number_of_nodes()} nodes (with TO)")
    print(f"Number of edges: {network.number_of_edges()} edges (with TO)")

    network_dir = config.directory.networks / name
    network_dir.mkdir(parents=True, exist_ok=True)
    write_network_json(network, network_dir)

    network = read_network_json(name)
    print(f"Size of the network: {network.number_of_nodes()} nodes")
    print(f"Number of edges: {network.number_of_edges()} edges")
    print(f"Number of connected components: {nx.number_connected_components(network)}")
    print(f"Average node degree: {np.mean([d for _, d in network.degree()])}")
    print(f"Average clustering coefficient: {nx.average_clustering(network)}")
    print(
        f"Modularity: {nx.algorithms.community.modularity(network, list(nx.connected_components(network)))}"
    )
