from __future__ import annotations

import networkx as nx
import py4cytoscape as py4c  # type: ignore[import-untyped]

from network.network import read_network_json


def prepare_network_for_cytoscape(
    graph: nx.Graph, min_component_size: int | None = None
) -> None:
    # Rename ID to NodeID
    for node, attrs in graph.nodes(data=True):
        if "ID" in attrs:
            attrs["NodeID"] = attrs.pop("ID")

    # Remove all components with size less than the 'threshold'
    if min_component_size is not None:
        components = list(nx.connected_components(graph))
        small_components = [c for c in components if len(c) < min_component_size]
        nodes_to_remove = [node for component in small_components for node in component]
        graph.remove_nodes_from(nodes_to_remove)

    # Add first order enzyme classes as node attribute
    high_level_ec: dict[str, set[str]] = {}
    for node, attr in graph.nodes(data=True):
        if isinstance(attr["ec"], str):
            ec_nums = attr["ec"].split(",")
            high_level_ecs = {ec.split(".")[0] for ec in ec_nums}
            high_level_ec[node] = high_level_ecs
        else:
            high_level_ec[node] = {"undefined"}

    # Use neighborhood to select first order enzyme class in case of multiple
    final_high_level_ec: dict[str, str] = {}
    for node, ecs in high_level_ec.items():
        if len(ecs) == 1:
            final_high_level_ec[node] = ecs.pop()
            continue

        neighbors = list(graph.neighbors(node))
        if len(neighbors) > 0:
            neighbor_ecs = set().union(*(high_level_ec[n] for n in neighbors))
            intersecting_ecs = ecs.intersection(neighbor_ecs)
            if len(intersecting_ecs) == 1:
                final_high_level_ec[node] = intersecting_ecs.pop()
            else:
                final_high_level_ec[node] = "ambiguous"
        else:
            final_high_level_ec[node] = "ambiguous"

    nx.set_node_attributes(graph, final_high_level_ec, "final_high_level_ec")


if __name__ == "__main__":
    network = "MBSNetwork"
    graph = read_network_json(network)

    # Add normalized RMSD attribute for visualization (edge attribute).
    for edge in graph.edges:
        rmsd = graph.edges[edge]["rmsd"]
        max_rmsd = 0.9
        rmsd_norm = 1 - (rmsd / max_rmsd)
        graph.edges[edge]["rmsd_norm"] = rmsd_norm

    prepare_network_for_cytoscape(graph, min_component_size=20)

    py4c.create_network_from_networkx(
        graph, title="MBSNetwork", collection="MBSNetworks"
    )
    py4c.set_node_size_default(45)
    py4c.set_node_shape_default("ellipse")
