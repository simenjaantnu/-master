from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import open3d  # type: ignore[import-untyped]
from config import config
import networkx as nx

if TYPE_CHECKING:
    from pathlib import Path

    from config import AlignmentConfig

    from alignment.alignment import MBSPointCloud


def get_loss_function(config: AlignmentConfig):
    match config.loss_function:
        case "tukey":
            return open3d.pipelines.registration.TukeyLoss(
                config.configuration_parameter
            )
        case "huber":
            return open3d.pipelines.registration.HuberLoss(
                config.configuration_parameter
            )
        case "gm":
            return open3d.pipelines.registration.GemanMcClureLoss(
                config.configuration_parameter
            )
        case "cauchy":
            return open3d.pipelines.registration.CauchyLoss(
                config.configuration_parameter
            )
        case "L1":
            return open3d.pipelines.registration.L1Loss()
        case "L2":
            return open3d.pipelines.registration.L2Loss()
    raise ValueError(f"Invalid loss function: {config.loss_function}")


def split_into_sublists(lst: list[Any], n: int):
    """Split 'lst' into 'n' (approximately) equally sized sublists."""
    sublist_size = math.ceil(len(lst) / n)
    return [lst[i : i + sublist_size] for i in range(0, len(lst), sublist_size)]


def interval_split(interval: tuple[int, int], n: int):
    """Split 'interval' into 'n' (approximately) equally sized intervals."""
    start, end = interval
    interval_length = end - start
    base_length = interval_length // n
    remainder = interval_length % n

    intervals: list[tuple[int, int]] = []
    current_start = start
    for i in range(n):
        # Add an extra one for the first 'remainder' intervals
        current_end = current_start + base_length + (1 if i < remainder else 0)
        intervals.append((current_start, current_end))
        current_start = current_end

    return intervals


def combination_within_interval(
    num_elements: int, subset_size: int, start: int, end: int
):
    """A generator function to yield combinations within a specified interval."""
    count = 0
    for combination in itertools.combinations(range(num_elements), subset_size):
        if start <= count < end:
            yield combination
        elif count >= end:
            break
        count += 1


def evaluate_alignment(
    source: MBSPointCloud, target: MBSPointCloud, rmsd_threshold: float
) -> float:
    """Calculate inlier RMSD of source point cloud to target point cloud.

    Ignores a fraction (rmsd_threshold) of the worst performing correspondences.
    """
    distances = np.asarray(source.compute_point_cloud_distance(target.point_cloud))

    if rmsd_threshold > 0:
        distances.sort()
        distances = distances[: -int(rmsd_threshold * len(distances))]
    return np.sqrt(np.mean(distances**2))


def get_alignment_matrix(rmsds: np.ndarray, pair_indices: list[tuple[int, int]]):
    """
    Construct a symmetric alignment matrix whose elements (i, j) denote the
    ICP alignment score of MBS i and j.

    Parameters
    ----------
    rmses : np.ndarray
        Array of the final alignment RMSE scores
    pair_indices : list[tuple]
        List of pairs of indices denoting the pairwise combinations of MBS
        structures in 'pcl_coords' to be aligned

    Returns
    ----------
    alignment_matrix : np.ndarray
        Symmetric alignment matrix (diagonal set to 1)
    """
    n = max(max(pair) for pair in pair_indices)
    alignment_matrix = np.identity(n + 1)
    for k, (i, j) in enumerate(pair_indices):
        alignment_matrix[i, j] = rmsds[k]
        alignment_matrix[j, i] = rmsds[k]
    return alignment_matrix


def save_rmsd_density(out_path: Path) -> None:
    bins = 1000
    range_min, range_max = 0.0, 2.5

    bin_edges = np.linspace(range_min, range_max, bins + 1)
    hist_counts = np.zeros(bins, dtype=np.int64)

    alignment_dir = config.directory.alignments

    for network_path in alignment_dir.glob("*.npy"):
        alignments = np.load(network_path, mmap_mode="r")
        rmsd_chunk = alignments[:, 2]
        hist, _ = np.histogram(rmsd_chunk, bins=bin_edges)
        hist_counts += hist

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_density = hist_counts / hist_counts.sum() / (bin_edges[1] - bin_edges[0])

    np.savez_compressed(
        out_path,
        bin_centers=bin_centers,
        hist_density=hist_density,
        range_min=range_min,
        range_max=range_max,
    )

def get_mbs_neighbors(
        network: nx.graph,
        node: nx.node
        
) -> list[MBSPointCloud]:
    
    """
    returns the neigbors of the given node
    """
    pass

if __name__ == "__main__":
    out_path = config.directory.analysis / "rmsd_density.npz"
    save_rmsd_density(out_path)
