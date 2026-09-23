from __future__ import annotations

import numpy as np
from alignment.alignment import PairwiseAligner
from config import config
from database import Session
from database.datamodel.models import MetalBindingSite
from scipy.integrate import simpson  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]
from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]
from sqlalchemy import select


def compute_bimodal_overlap_gmm(rmsd_values: np.ndarray) -> float:
    """Fit a 2-component GMM to the data, extract component parameters,
    and compute the area of overlap between the two Gaussians."""
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    gmm.fit(rmsd_values.reshape(-1, 1))

    # Extract means and stds, ordered by mean
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    idx = np.argsort(means)
    mu1, mu2 = means[idx]
    sigma1, sigma2 = stds[idx]

    # Define x-range using ±4σ around each mean
    x_min = min(mu1 - 4 * sigma1, mu2 - 4 * sigma2)
    x_max = max(mu1 + 4 * sigma1, mu2 + 4 * sigma2)
    x = np.linspace(x_min, x_max, 1000)

    # Compute PDFs
    pdf1 = norm.pdf(x, mu1, sigma1)
    pdf2 = norm.pdf(x, mu2, sigma2)

    # Overlap = area under pointwise min of the PDFs
    overlap = np.minimum(pdf1, pdf2)
    overlap_area = simpson(overlap, x=x)

    return overlap_area


def bimodal_overlap(ranges: list[tuple[float, float]], n_pairs: int) -> np.ndarray:
    """Compute mode overlap between modes of bimodal RMSD distributions for a given
    number of MBS point cloud pairs within specified ranges."""
    alignments_path = config.directory.alignments / "final_alignments_node_2.npy"
    alignments = np.load(alignments_path)

    pairs_by_range: dict[float, list[tuple[int, int]]] = {low: [] for low, _ in ranges}
    all_mbs_ids = set()

    # Iterate over alignments until we have n_pairs for every range.
    indices = np.arange(len(alignments))
    np.random.shuffle(indices)

    for i in indices:
        source_id, target_id, rmsd = alignments[i]
        if all(len(pairs_by_range[low]) >= n_pairs for low, _ in ranges):
            break

        for low, high in ranges:
            if low <= rmsd < high and len(pairs_by_range[low]) < n_pairs:
                pairs_by_range[low].append((source_id, target_id))
                all_mbs_ids.update([source_id, target_id])
                break

    # Deduplicate and index MBS IDs.
    ordered_mbs_ids = sorted(all_mbs_ids)
    mbs_id_to_idx = {mbs_id: i for i, mbs_id in enumerate(ordered_mbs_ids)}

    # Re-encode all pairs as index pairs into ordered_mbs_ids.
    ordered_pairs = []
    for low, _ in ranges:
        for src_id, tgt_id in pairs_by_range[low]:
            src_idx = mbs_id_to_idx[src_id]
            tgt_idx = mbs_id_to_idx[tgt_id]
            ordered_pairs.append((src_idx, tgt_idx))

    alignments = compute_rmsd_distribution(ordered_mbs_ids, ordered_pairs)

    # Compute GMM overlap per range.
    mode_overlaps = np.zeros(len(ranges))
    for i in range(len(ranges)):
        start = i * n_pairs
        end = start + n_pairs
        rmsd_values = alignments[start:end, 2]
        mode_overlaps[i] = compute_bimodal_overlap_gmm(rmsd_values)

    path = config.directory.analysis / "bimodal_overlap.npy"
    with path.open("wb") as f:
        np.save(f, mode_overlaps)

    return mode_overlaps


def compute_rmsd_distribution(
    mbs_ids: list[int], pairs: list[tuple[int, int]]
) -> np.ndarray:
    """Calculate RMSD distributions for pairs of MBS point clouds."""
    with Session() as session:
        mbss = (
            session.execute(
                select(MetalBindingSite).where(MetalBindingSite.id.in_(mbs_ids))
            )
            .scalars()
            .all()
        )

    # Align all pairs of MBS point clouds.
    mbs_dict = {site.id: site for site in mbss}
    sorted_mbss = [mbs_dict[site_id] for site_id in mbs_ids]
    aligner = PairwiseAligner(sorted_mbss, config.alignment, pair_indices=pairs)
    alignments = aligner.align()

    return alignments


if __name__ == "__main__":
    ranges = [
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 1.0),
        (1.0, 1.1),
        (1.1, 1.2),
        (1.2, 1.3),
        (1.3, 1.4),
        (1.4, 1.5),
        (1.5, 1.6),
        (1.6, 1.7),
        (1.7, 1.8),
    ]
    n_pairs = 10000
    gmm_overlap_areas = bimodal_overlap(ranges, n_pairs)
    print(gmm_overlap_areas)

