from __future__ import annotations

import copy
import logging
import os
import time
from typing import TYPE_CHECKING, Any
from config import config

import numpy as np
import open3d  # type: ignore[import-untyped]
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from logger import configure_queue_listener, configure_worker_logger
import networkx as nx

from alignment.util import (
    combination_within_interval,
    evaluate_alignment,
    get_loss_function,
    interval_split,
    split_into_sublists,
    get_mbs_neighbors,
)

if TYPE_CHECKING:
    from queue import Queue

    from config import AlignmentConfig
    from database.datamodel.models import MetalBindingSite

logger = logging.getLogger(__name__)


class MBSPointCloud:
    """A wrapper class of Open3d PointCloud with added database ID and utility methods.

    Drop-in replacement for Open3d PointCloud as the pybind11 bindings
    do not support inheritance (see https://github.com/isl-org/Open3D/issues/572).
    """

    def __init__(self, db_id: int):
        self.db_id = db_id
        self.point_cloud = open3d.geometry.PointCloud()

    def set_points(self, points: open3d.utility.Vector3dVector):
        self.point_cloud.points = points

    def rotate_randomly(self):
        """Rotate point cloud randomly around its centroid."""
        R = self._random_rotation_matrix()
        self.random_rotation = R
        self.point_cloud.rotate(R)
        T = np.eye(4)
        T[:3, :3] = R
        return T

    def _random_rotation_matrix(self):
        return self.point_cloud.get_rotation_matrix_from_xyz(
            np.random.rand(3, 1) * 2 * np.pi
        )

    def center_point_cloud(self):
        """Center point cloud at its centroid."""
        points = np.asarray(self.point_cloud.points)
        centroid = [sum(x) / len(x) for x in zip(*points)]
        self.set_points(open3d.utility.Vector3dVector(points - centroid))
        self.centroid = np.array(centroid)

    @classmethod
    def from_mbs(cls, mbs: MetalBindingSite):
        point_cloud_wrapper = cls(mbs.id)
        point_cloud_wrapper.set_points(
            open3d.utility.Vector3dVector(
                np.column_stack((mbs.x_coords, mbs.y_coords, mbs.z_coords))
            )
        )
        point_cloud_wrapper.center_point_cloud()
        point_cloud_wrapper.rotate_randomly()
        point_cloud_wrapper.point_cloud.estimate_normals()
        point_cloud_wrapper.point_cloud.estimate_covariances()
        return point_cloud_wrapper

    def __getattr__(self, name: str):
        try:
            return getattr(self.point_cloud, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self.point_cloud).__name__}' object has no attribute '{name}'"
            )

    def __deepcopy__(self, memo: dict[int, Any]):
        new_copy = type(self)(self.db_id)
        new_copy.point_cloud = copy.deepcopy(self.point_cloud, memo)
        return new_copy


class PairwiseAligner:
    """Helper class to initialize and run all-to-all pairwise point cloud alignments."""

    def __init__(
        self,
        mbss: list[MetalBindingSite],
        config: AlignmentConfig,
        root_logger: logging.Logger | None = None,
        pair_indices: list[tuple[int, int]] | None = None,
    ):
        self.mbss = mbss
        self.config = config
        self.root_logger = root_logger
        self.pair_indices = pair_indices

    def save_alignment(self, alignment_rmsds: np.ndarray):
        assert self.config.path is not None
        path = self.config.path / "final_alignments.npy"
        with path.open("wb") as f:
            np.save(f, alignment_rmsds)

    def align(
        self,
        n_jobs: int | None = None,
        save_flag: bool = False,
        resume: bool = False,
    ) -> np.ndarray:
        """Run all-to-all pairwise point cloud alignment.

        Number of jobs (n_jobs) has the same meaning as in joblib.Parallel, that is for negative
        values, the number of jobs is defined as os.cpu_count() + 1 + n_jobs. Note that this option
        overrides the n_jobs setting in the configuration.
        """
        if n_jobs is None:
            n_jobs = self.config.n_jobs

        # Divide alignments into intervals for parallel processing
        if n_jobs < 0:
            workers = os.cpu_count() + 1 + n_jobs  # type: ignore[operator]
        else:
            workers = n_jobs

        if self.root_logger is not None:
            queue, listener = configure_queue_listener(self.root_logger)
            log_level = self.root_logger.getEffectiveLevel()
            listener.start()
        else:
            queue = None
            listener = None
            log_level = None

        all_alignments: np.ndarray
        if self.pair_indices is None:
            n_mbss = len(self.mbss)
            combinations = n_mbss * (n_mbss - 1) // 2
            intervals = interval_split((0, combinations), workers)

            if workers == 1:
                all_alignments = pairwise_alignment(
                    self.mbss, intervals[0], 0, self.config, save_flag, resume
                )
            else:
                job_alignment_rmsds = Parallel(n_jobs=n_jobs)(
                    delayed(pairwise_alignment)(
                        self.mbss,
                        interval,
                        worker,
                        self.config,
                        save_flag,
                        resume,
                        log_queue=queue,
                        log_level=log_level,
                    )
                    for interval, worker in zip(intervals, range(workers))
                )
                all_alignments = np.vstack(job_alignment_rmsds)
        else:
            # Split pair indices into sublists for each worker
            pair_indices_split = split_into_sublists(self.pair_indices, workers)

            if len(pair_indices_split) != workers:
                raise ValueError("Number of workers must match the number of sublists.")

            job_alignment_rmsds = Parallel(n_jobs=n_jobs)(
                delayed(pairwise_alignment)(
                    self.mbss,
                    None,
                    worker,
                    self.config,
                    save_flag,
                    resume,
                    sub_pair_indices,
                    log_queue=queue,
                    log_level=log_level,
                )
                for worker, sub_pair_indices in zip(range(workers), pair_indices_split)
            )
            all_alignments = np.vstack(job_alignment_rmsds)

        if listener is not None:
            listener.stop()

        if save_flag:
            self.save_alignment(all_alignments)
        return all_alignments


def configure_icp_params(config: AlignmentConfig) -> None:
    """Derive config's open3d ICP objects (transformation_estimation, criteria_*,
    finetune_number) from its plain settings (loss_function, max_coarse_iterations,
    etc.). align_point_clouds needs these set; pairwise_alignment does this once
    up front for a whole batch, but any other caller must do it too.
    """
    loss_function = get_loss_function(config)
    config.transformation_estimation = (
        open3d.pipelines.registration.TransformationEstimationForGeneralizedICP(
            1.0, loss_function
        )
    )
    config.criteria_coarse = open3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=config.max_coarse_iterations
    )
    config.criteria_finetune = open3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=config.max_finetune_iterations
    )
    config.finetune_number = int(config.coarse_alignments * config.top_ranking_fraction)


def align_point_clouds(
    source: MBSPointCloud,
    target: MBSPointCloud,
    config: AlignmentConfig,
    return_transformation: bool = False,
):
    """Align two point clouds using ICP and return the RMSD and, optionally, the transformation."""
    if config.transformation_estimation is None:
        configure_icp_params(config)

    best_initial_rmsds = np.ones(config.finetune_number) * 1000.0
    best_initial_targets: list[MBSPointCloud] = [
        None for _ in range(config.finetune_number)
    ]  # type: ignore
    best_initial_transformations = [None for _ in range(config.finetune_number)]
    best_initial_rotations = [None for _ in range(config.finetune_number)]

    # Coarse alignment
    rotated_target = copy.deepcopy(target)
    for k in range(config.coarse_alignments):
        rotated_target = copy.deepcopy(target)
        R = rotated_target.rotate_randomly()

        coarse_alignment = open3d.pipelines.registration.registration_generalized_icp(
            source.point_cloud,
            rotated_target.point_cloud,
            config.max_correspondence_distance,
            estimation_method=config.transformation_estimation,
            criteria=config.criteria_coarse,
        )

        # Update top rankings
        n_correspondences = len(np.asarray(coarse_alignment.correspondence_set))
        best_idx = np.argmax(best_initial_rmsds)
        if (
            n_correspondences > 0
            and coarse_alignment.inlier_rmse < best_initial_rmsds[best_idx]
        ):
            best_initial_rmsds[best_idx] = coarse_alignment.inlier_rmse
            best_initial_targets[best_idx] = copy.deepcopy(rotated_target)
            best_initial_transformations[best_idx] = coarse_alignment.transformation
            best_initial_rotations[best_idx] = R

    # Fine-tune best alignments
    final_rmsd = 1000.0
    best_full_transformation = None
    for initial_target, init_transform, init_rotation in zip(
        best_initial_targets, best_initial_transformations, best_initial_rotations
    ):
        transformed_source = copy.deepcopy(source)
        transformed_source.transform(init_transform)

        finetuned_alignment = (
            open3d.pipelines.registration.registration_generalized_icp(
                transformed_source.point_cloud,
                initial_target.point_cloud,
                config.max_correspondence_distance,
                estimation_method=config.transformation_estimation,
                criteria=config.criteria_finetune,
            )
        )

        transformed_source.transform(finetuned_alignment.transformation)
        rmsd = evaluate_alignment(
            transformed_source, initial_target, config.rmsd_ignore_threshold
        )
        if rmsd < final_rmsd:
            final_rmsd = rmsd

            # Combine coarse + fine + initial rotation
            T_coarse = init_transform
            T_fine = finetuned_alignment.transformation
            T_combined = T_fine @ T_coarse
            T_full = init_rotation @ T_combined
            best_full_transformation = T_full

    if return_transformation:
        return final_rmsd, best_full_transformation
    return final_rmsd


def simulation_alignment(
        simulation_frames: list[MBSPointCloud],
        target:MBSPointCloud,
        config:AlignmentConfig,
        init_transform=None,

):
    """Perform local alignment on all simulation frames 
    in relation to the given target site"""


    """Pseudocode:
    - retrieve neighboring sites from MBS_network
    - Do thorough simulation of the first alignment if not given.
    - Save the last transformation. 
    - Do a simple alignment on the next fram, using the last transformation as a target.
    - Return the original
    """

    if init_transform is None:
        init_rmsd, init_transform = align_point_clouds(
        simulation_frames[0],
        target,
        config,
        True)
    else:
        init_rmsd = evaluate_alignment()
    # Fine-tune best alignments
    final_rmsd = init_rmsd
    best_full_transformation = init_transform
    print("amount of frames to compare: ",len(simulation_frames))
    for i in range(1, len(simulation_frames)):
        source = simulation_frames[i]
        transformed_source = copy.deepcopy(source)
        transformed_source.transform(init_transform)

        finetuned_alignment = (
            open3d.pipelines.registration.registration_generalized_icp(
                transformed_source.point_cloud,
                target.point_cloud,
                config.max_correspondence_distance,
                estimation_method=config.transformation_estimation,
                criteria=config.criteria_finetune,
            )
        )

        transformed_source.transform(finetuned_alignment.transformation)
        rmsd = evaluate_alignment(
            transformed_source, target, config.rmsd_ignore_threshold
        )
        if rmsd < final_rmsd:
            final_rmsd = rmsd
            # Combine coarse + fine + initial rotation
            T_coarse = init_transform
            T_full = T_coarse @ finetuned_alignment.transformation
            best_full_transformation = T_full

        
    return final_rmsd

def simulation_alignment_from_node(
        network: nx.Graph,
        node: int,
        config: AlignmentConfig,
        threshold: float

):
        """
        Re-evaluates the network at the given node, using a short simulation. 
        Only re-evaluates the immediate neighbors of the given node.
        """
        node = network.nodes[node]
        dp_id = node["pdb_entry"]

def network_based_pairwise_alignment(
    network: nx.Graph,
    mbss: list[MetalBindingSite],
    simulation_frames_by_node: dict[int, list[MBSPointCloud]],
    config: AlignmentConfig,
    threshold: float,
) -> nx.Graph:
    """Re-evaluate and prune network edges using simulation-based alignment.

    For each node that has simulation frames available, re-evaluate the rmsd
    of its edges that currently exceed `threshold` via `simulation_alignment`,
    keeping the lower of the original and re-evaluated rmsd. Afterwards,
    remove all edges whose rmsd is still above `threshold`. Checks both ways,
      i.e. both simulation from node u to target v, and node v to target u.
    """

    """
    Pseudocode:
    takes a series of 
    """
    mbs_by_id = {mbs.id: mbs for mbs in mbss}

    for node in network.nodes():
        simulation_frames = simulation_frames_by_node.get(node) #Replace this with 
        if not simulation_frames:
            continue

        for neighbor in list(network.neighbors(node)):
            edge_attrs = network.edges[node, neighbor]
            og_rmsd = edge_attrs.get("rmsd")

            if og_rmsd is None or og_rmsd <= threshold:
                continue  # already a good alignment, no need to re-evaluate

            target = MBSPointCloud.from_mbs(mbs_by_id[neighbor])
            rmsd, _ = simulation_alignment(simulation_frames, target, config)

            if rmsd < og_rmsd:
                edge_attrs["rmsd"] = rmsd

    edges_to_remove = [
        (u, v)
        for u, v, rmsd in network.edges(data="rmsd")
        if rmsd is not None and rmsd > threshold
    ]
    network.remove_edges_from(edges_to_remove)

    return network


def get_nodes_for_realignment():
    """Searches for the best nodes to do additional simulation and steps"""
    

def pairwise_alignment(
    mbss: list[MetalBindingSite],
    interval: tuple[int, int] | None,
    worker: int,
    config: AlignmentConfig,
    save_flag: bool,
    resume: bool,
    pair_indices: list[tuple[int, int]] | None = None,
    log_queue: Queue | None = None,
    log_level: int | None = None,
):
    """Perform all-to-all pairwise ICP alignments of the MBSs."""
    # Set up worker logger
    if log_queue is not None:
        worker_logger = configure_worker_logger(log_queue, log_level)
    else:
        worker_logger = None

    # Instantiated here as Open3d objects are not serializable
    point_clouds = [MBSPointCloud.from_mbs(mbs) for mbs in mbss]

    # Set up the ICP registration parameters
    worker_path = config.path / f"alignment-worker-{worker}.npz"
    configure_icp_params(config)

    # Determine pairwise combinations
    if pair_indices is not None:
        combinations = len(pair_indices) + 1
        pairwise_combinations = pair_indices
    elif interval is not None:
        start = interval[0]
        end = interval[1]
        combinations = end - start + 1
        pairwise_combinations = combination_within_interval(
            len(point_clouds), 2, start, end
        )
    else:
        raise ValueError("Either interval or pair_indices must be provided.")

    # Initialize RMSD matrix
    if resume:
        loaded_data = np.load(worker_path)
        alignment_rmsds = loaded_data["alignment_rmsds"]
        rmsd_idx = loaded_data["rmsd_idx"]
    else:
        alignment_rmsds = np.zeros((combinations - 1, 3))
        rmsd_idx = 0

    # Run the pairwise alignments
    logsave_counter = 0
    start_time = time.time()
    for counter, (i, j) in enumerate(pairwise_combinations):
        logsave_counter += 1
        if counter < rmsd_idx:
            continue

        if worker_logger and worker == 0 and logsave_counter % config.log_divisor == 0:
            end_time = time.time()
            worker_logger.info(
                f"Worker {worker}: {logsave_counter} / {combinations} ({(end_time - start_time) / config.log_divisor:.5f} s/alignment)"
            )
            start_time = time.time()

        source = point_clouds[i]
        target = point_clouds[j]
        final_rmsd = align_point_clouds(source, target, config)
        alignment_rmsds[rmsd_idx, :] = [source.db_id, target.db_id, final_rmsd]
        rmsd_idx += 1

        if save_flag and logsave_counter % config.save_divisor == 0:
            np.savez(worker_path, alignment_rmsds=alignment_rmsds, rmsd_idx=rmsd_idx)
    return alignment_rmsds

if __name__=="__main__":
    

    import simulation.simulation as sim
    print("Starting main", flush=True)

    start_time = time.perf_counter()

    test_enzyme = "4M9E"
    ref_enzyme = "3VK6"
    sim_frames_all = sim.simulate_structure(test_enzyme)
    ref_MDS_all = sim.get_static_mbs(ref_enzyme)
    end_sim_time = time.perf_counter()
    for site1, sim_frames in sim_frames_all.items():
        for site2, ref_MDS in ref_MDS_all.items():

            rmsd_og = align_point_clouds(sim_frames[0],ref_MDS,config.alignment)
            rmsd_sim = simulation_alignment(sim_frames,ref_MDS,config.alignment)
            print()
            print(f"comparing site {site1} and site {site2}")
            print(f"original rmsd: {rmsd_og}")
            print(f"reevaluated rmsd: {rmsd_sim}")
            print()


    end_time = time.perf_counter()
    sim_time = end_sim_time - start_time
    comp_time = end_time - end_sim_time
    tot_time = end_time - start_time

    print("time used: ",tot_time)
    print("time used for simulation: ", sim_time)
    print("time used for comparison: ", comp_time)

    amount_of_comparisons = len(ref_MDS_all)*len(sim_frames_all)/2
    print(f"Total amount of comparisons: {amount_of_comparisons}")