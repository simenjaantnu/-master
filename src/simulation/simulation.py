from __future__ import annotations

import asyncio
import gzip
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d  # type: ignore[import-untyped]
import openmm  # type: ignore[import-untyped]
from openmm import app, unit  # type: ignore[import-untyped]
from pdbfixer import PDBFixer  # type: ignore[import-untyped]

from alignment.alignment import MBSPointCloud
from config import config
from preprocessing.util import download_cif

logger = logging.getLogger(__name__)

AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "CYM", "ASH", "GLH", "LYN",
}
# Terminal capping groups (common in NMR peptide structures, e.g. 1ZNF). These
# are covalently bonded to the chain and must be kept, or the neighboring
# residue is left with a dangling external bond and fails template matching.
CAPPING_GROUPS = {"ACE", "NME", "NH2"}
PROTEIN_RESIDUES = AMINO_ACIDS | CAPPING_GROUPS
COORDINATING_ELEMENTS = {"N", "O", "S", "SE"}

# OpenMM works in nm/kJ; the MBS database stores Angstrom (Biopython convention).
KCAL_PER_ANGSTROM_SQ = 418.4  # -> kJ/mol/nm^2


@dataclass
class SimulationConfig:
    """Parameters for a short, geometry-preserving metalloprotein simulation."""

    coordination_cutoff: float = 2.8  # Angstrom
    metal_ligand_k: float = 200.0  # kcal/mol/A^2
    ligand_ligand_k: float = 50.0  # kcal/mol/A^2
    temperature: float = 300.0  # K
    friction: float = 1.0  # 1/ps
    timestep: float = 0.002  # ps
    equilibration_steps: int = 2_500  # 5 ps
    production_steps: int = 5_000  # 10 ps
    frame_interval: int = 50  # 1 ps -> 100 frames
    solvent_padding: float = 1.0  # nm
    ionic_strength: float = 0.15  # molar
    ph: float = 7.0
    explicit_solvent: bool = True
    platform: str | None = None
    output_dir: Path = field(
        default_factory=lambda: config.directory.structures.parent / "simulations"
    )


class MBSFrameReporter:
    """OpenMM reporter that records only the MBS atoms of each metal site.

    Storing the fixed atom selection per frame (rather than re-deriving the
    nearest atoms each frame) keeps point correspondence consistent across the
    trajectory, which downstream ICP alignment depends on.
    """

    def __init__(self, site_indices: dict[str, np.ndarray], interval: int):
        self._site_indices = site_indices
        self._interval = interval
        self.frames: dict[str, list[np.ndarray]] = {key: [] for key in site_indices}

    def describeNextReport(self, simulation):  # noqa: N802 (OpenMM API)
        steps = self._interval - simulation.currentStep % self._interval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        for key, indices in self._site_indices.items():
            self.frames[key].append(np.asarray(positions[indices], dtype=float))


def fetch_structure(entry_id: str, assembly_id: str = "1") -> Path:
    """Fetch a biological assembly from RCSB, reusing the preprocessing downloader."""
    path = asyncio.run(download_cif(entry_id, assembly_id))
    if path is None:
        raise RuntimeError(f"Could not download {entry_id} assembly {assembly_id}.")
    return path


METAL_SYMBOLS = {ligand.strip().upper() for ligand in config.dataset.ligands}


def _is_metal(atom) -> bool:
    return atom.element is not None and atom.element.symbol.upper() in METAL_SYMBOLS


def prepare_system(cif_path: Path, sim_config: SimulationConfig):
    """Protonate, clean and solvate the assembly, returning an OpenMM system."""
    with tempfile.NamedTemporaryFile(suffix=".cif") as temp_file:
        with gzip.open(cif_path) as zip_file:
            shutil.copyfileobj(zip_file, temp_file)  # type: ignore[misc]
        temp_file.flush()
        with Path(temp_file.name).open() as handle:
            fixer = PDBFixer(pdbxfile=handle)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    modeller = app.Modeller(fixer.topology, fixer.positions)

    # Keep protein (including terminal caps like ACE/NH2) and the target metals;
    # waters are rebuilt during solvation.
    to_delete = [
        residue
        for residue in modeller.topology.residues()
        if residue.name.upper() not in PROTEIN_RESIDUES
        and not any(_is_metal(atom) for atom in residue.atoms())
    ]
    modeller.delete(to_delete)

    if not any(_is_metal(atom) for atom in modeller.topology.atoms()):
        raise ValueError(f"No target metal found in {cif_path.name}.")

    water_model = "amber14/tip3pfb.xml"
    forcefield = app.ForceField("amber14-all.xml", water_model)
    modeller.addHydrogens(forcefield, pH=sim_config.ph)

    if sim_config.explicit_solvent:
        modeller.addSolvent(
            forcefield,
            padding=sim_config.solvent_padding * unit.nanometer,
            ionicStrength=sim_config.ionic_strength * unit.molar,
        )

    try:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME if sim_config.explicit_solvent else app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
        )
    except ValueError as error:
        missing = sorted(
            {
                atom.element.symbol
                for atom in modeller.topology.atoms()
                if _is_metal(atom)
            }
        )
        raise ValueError(
            f"Force field has no template for one of {missing}. Exotic transition "
            f"metals (Mo, W, V) lack standard AMBER ion parameters and need "
            f"explicit parameters supplied. Original error: {error}"
        ) from error
    print("system prepared")
    return modeller, system, forcefield


def find_coordination_shells(
    topology, positions: np.ndarray, cutoff: float
) -> dict[int, list[int]]:
    """Map each metal atom index to the protein heavy atoms coordinating it."""
    candidates = [
        atom
        for atom in topology.atoms()
        if atom.element is not None
        and atom.element.symbol.upper() in COORDINATING_ELEMENTS
        and atom.residue.name.upper() in AMINO_ACIDS
    ]
    candidate_indices = np.array([atom.index for atom in candidates], dtype=int)

    shells: dict[int, list[int]] = {}
    for atom in topology.atoms():
        if not _is_metal(atom):
            continue
        distances = np.linalg.norm(
            positions[candidate_indices] - positions[atom.index], axis=1
        )
        shells[atom.index] = candidate_indices[distances <= cutoff].tolist()

    return shells


def add_coordination_restraints(
    system,
    positions: np.ndarray,
    shells: dict[int, list[int]],
    sim_config: SimulationConfig,
) -> int:
    """Restrain metal-ligand and ligand-ligand distances to their crystal values.

    Ligand-ligand restraints are what actually hold the coordination *geometry*:
    metal-ligand distances alone leave the shell free to distort angularly
    (e.g. tetrahedral collapsing toward square planar).
    """
    force = openmm.CustomBondForce("0.5*k*(r-r0)^2")
    force.addPerBondParameter("k")
    force.addPerBondParameter("r0")

    metal_k = sim_config.metal_ligand_k * KCAL_PER_ANGSTROM_SQ
    ligand_k = sim_config.ligand_ligand_k * KCAL_PER_ANGSTROM_SQ
    n_restraints = 0

    for metal_index, ligand_indices in shells.items():
        for ligand_index in ligand_indices:
            r0 = np.linalg.norm(positions[ligand_index] - positions[metal_index]) / 10.0
            force.addBond(metal_index, ligand_index, [metal_k, r0])
            n_restraints += 1

        for i, first in enumerate(ligand_indices):
            for second in ligand_indices[i + 1 :]:
                r0 = np.linalg.norm(positions[second] - positions[first]) / 10.0
                force.addBond(first, second, [ligand_k, r0])
                n_restraints += 1

    system.addForce(force)
    return n_restraints


def select_mbs_atoms(
    topology, positions: np.ndarray, metal_index: int, n_atoms: int
) -> np.ndarray:
    """Pick the n closest protein heavy atoms to a metal, mirroring get_mbs()."""
    candidates = np.array(
        [
            atom.index
            for atom in topology.atoms()
            if atom.element is not None
            and atom.element.symbol != "H"
            and atom.residue.name.upper() in AMINO_ACIDS
        ],
        dtype=int,
    )
    distances = np.linalg.norm(
        positions[candidates] - positions[metal_index], axis=1
    )
    closest = np.argpartition(distances, min(n_atoms, len(candidates) - 1))[:n_atoms]
    return candidates[closest]


def get_static_mbs(
    entry_id: str,
    assembly_id: str = "1",
    coordination_cutoff: float = 2.8,
    db_id: int | None = None,
) -> dict[str, MBSPointCloud]:
    """Get every metal binding site's point cloud straight from the crystal structure.

    No MD simulation - fetches the assembly, cleans it with PDBFixer, and takes
    the n closest protein heavy atoms around each metal. Returns one entry per
    metal center found, keyed by site_key (e.g. "site_1523"), the same
    convention used by simulate_structure.

    If db_id is not given, each site's id is resolved from the database by
    matching its crystal-frame metal position against stored
    MetalBindingSite.ligand_coord values (see resolve_site_db_id) - this
    requires the entry to already be deposited in the database. An explicit
    db_id is only accepted when the assembly has exactly one metal site.
    """
    cif_path = fetch_structure(entry_id, assembly_id)

    with tempfile.NamedTemporaryFile(suffix=".cif") as temp_file:
        with gzip.open(cif_path) as zip_file:
            shutil.copyfileobj(zip_file, temp_file)  # type: ignore[misc]
        temp_file.flush()
        with Path(temp_file.name).open() as handle:
            fixer = PDBFixer(pdbxfile=handle)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    modeller = app.Modeller(fixer.topology, fixer.positions)
    to_delete = [
        residue
        for residue in modeller.topology.residues()
        if residue.name.upper() not in PROTEIN_RESIDUES
        and not any(_is_metal(atom) for atom in residue.atoms())
    ]
    modeller.delete(to_delete)

    positions = np.asarray(
        modeller.positions.value_in_unit(unit.angstrom), dtype=float
    )
    shells = find_coordination_shells(modeller.topology, positions, coordination_cutoff)

    if db_id is not None and len(shells) != 1:
        raise ValueError(
            f"An explicit db_id was given but {entry_id} assembly {assembly_id} has "
            f"{len(shells)} metal sites; db_id only applies to a single-site assembly."
        )

    point_clouds: dict[str, MBSPointCloud] = {}
    for metal_index in shells:
        atom_indices = select_mbs_atoms(
            modeller.topology, positions, metal_index, config.dataset.atoms
        )
        site_db_id = (
            db_id
            if db_id is not None
            else resolve_site_db_id(entry_id, assembly_id, positions[metal_index])
        )

        point_cloud = MBSPointCloud(site_db_id)
        point_cloud.set_points(open3d.utility.Vector3dVector(positions[atom_indices]))
        point_cloud.center_point_cloud()
        point_cloud.point_cloud.estimate_normals()
        point_cloud.point_cloud.estimate_covariances()
        point_clouds[f"site_{metal_index}"] = point_cloud

    return point_clouds


def resolve_site_db_id(
    entry_id: str, assembly_id: str, metal_position: np.ndarray, tolerance: float = 1.0
) -> int:
    """Match a simulated metal's crystal-frame position to its MetalBindingSite.id.

    PDBFixer/solvation don't move existing heavy atoms, so the metal's position
    at simulation start should still match the ligand_coord recorded in the
    database for the same assembly (to within a small tolerance, in Angstrom).
    """
    # Imported lazily: database connects to Postgres at import time, and most
    # of this module (e.g. get_static_mbs with an explicit db_id) has no need
    # for a live DB connection just to be imported.
    from sqlalchemy import select

    from database import Session
    from database.datamodel.models import Assembly

    with Session() as session:
        assembly = session.execute(
            select(Assembly).where(
                Assembly.entry_pdb_id == entry_id, Assembly.assembly_id == assembly_id
            )
        ).scalar_one()
        candidates = list(assembly.metal_binding_sites or [])

    if not candidates:
        raise ValueError(
            f"No metal binding sites in the database for {entry_id} assembly {assembly_id}."
        )

    distances = [
        np.linalg.norm(np.array(mbs.ligand_coord) - metal_position)
        for mbs in candidates
    ]
    best = int(np.argmin(distances))
    if distances[best] > tolerance:
        raise ValueError(
            f"No confident MetalBindingSite match for {entry_id} assembly "
            f"{assembly_id}: closest candidate is {distances[best]:.2f} A away "
            f"(tolerance {tolerance})."
        )
    return candidates[best].id


def simulate_structure(
    entry_id: str,
    assembly_id: str = "1",
    sim_config: SimulationConfig | None = None,
) -> dict[str, list[MBSPointCloud]]:
    """Fetch, simulate and save MBS trajectories for one biological assembly.

    Frames are always saved to disk (.npz), and every metal site found is
    always loaded and returned as a dict[site_key, list[MBSPointCloud]], one
    entry per metal center. Each site's db_id is resolved from the database
    by matching its simulated metal's crystal-frame position against stored
    MetalBindingSite.ligand_coord values.

    return_frames_immediately currently has no effect on the return value -
    kept for call-site compatibility.
    """
    sim_config = sim_config or SimulationConfig()
    output_dir = sim_config.output_dir / f"{entry_id}_assembly{assembly_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_path = fetch_structure(entry_id, assembly_id)
    modeller, system, _ = prepare_system(cif_path, sim_config)

    positions = (
        np.asarray(modeller.positions.value_in_unit(unit.angstrom), dtype=float)
    )
    shells = find_coordination_shells(
        modeller.topology, positions, sim_config.coordination_cutoff
    )
    n_restraints = add_coordination_restraints(system, positions, shells, sim_config)
    logger.info(
        f"{entry_id}: {len(shells)} metal site(s), {n_restraints} geometry restraints."
    )

    site_indices = {
        f"site_{metal_index}": select_mbs_atoms(
            modeller.topology, positions, metal_index, config.dataset.atoms
        )
        for metal_index in shells
    }

    integrator = openmm.LangevinMiddleIntegrator(
        sim_config.temperature * unit.kelvin,
        sim_config.friction / unit.picosecond,
        sim_config.timestep * unit.picoseconds,
    )
    platform = (
        openmm.Platform.getPlatformByName(sim_config.platform)
        if sim_config.platform
        else None
    )
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    logger.info(f"{entry_id}: minimizing...")
    simulation.minimizeEnergy()

    with (output_dir / "topology.pdb").open("w") as handle:
        app.PDBFile.writeFile(
            modeller.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            handle,
        )

    simulation.context.setVelocitiesToTemperature(sim_config.temperature * unit.kelvin)
    logger.info(f"{entry_id}: equilibrating...")
    simulation.step(sim_config.equilibration_steps)

    reporter = MBSFrameReporter(site_indices, sim_config.frame_interval)
    simulation.reporters.append(reporter)
    simulation.reporters.append(
        app.DCDReporter(str(output_dir / "trajectory.dcd"), sim_config.frame_interval)
    )

    logger.info(f"{entry_id}: running production...")
    simulation.step(sim_config.production_steps)
    frames_path = output_dir / "mbs_frames.npz"
    ordered_site_keys = sorted(site_indices)
    np.savez_compressed(
        frames_path,
        **{key: np.asarray(value) for key, value in reporter.frames.items()},
        atom_indices=np.array(
            [site_indices[key] for key in ordered_site_keys], dtype=int
        ),
        site_keys=np.array(ordered_site_keys),
        # Crystal-frame metal positions, ordered to match site_keys, so a
        # simulated site can later be matched back to its MetalBindingSite.id.
        metal_positions=np.array(
            [positions[int(key.removeprefix("site_"))] for key in ordered_site_keys],
            dtype=float,
        ),
    )
    logger.info(f"{entry_id}: saved frames to {frames_path}")

    def _db_id_for(site_key: str) -> int:
        metal_index = int(site_key.removeprefix("site_"))
        return resolve_site_db_id(entry_id, assembly_id, positions[metal_index])

    return {
        site_key: load_simulation_frames(frames_path, site_key, _db_id_for(site_key))
        for site_key in ordered_site_keys
    }


def load_simulation_frames(
    frames_path: Path, site_key: str, db_id: int
) -> list[MBSPointCloud]:
    """Load saved frames as MBSPointClouds sharing a common reference frame.

    Deliberately avoids MBSPointCloud.from_mbs: that random-rotates each cloud,
    which breaks simulation_alignment's assumption that every frame can be
    placed by the same initial transformation.
    """
    data = np.load(frames_path)
    frames = data[site_key]

    point_clouds: list[MBSPointCloud] = []
    for frame in frames:
        point_cloud = MBSPointCloud(db_id)
        point_cloud.set_points(open3d.utility.Vector3dVector(frame))
        point_cloud.center_point_cloud()
        point_cloud.point_cloud.estimate_normals()
        point_cloud.point_cloud.estimate_covariances()
        point_clouds.append(point_cloud)

    return point_clouds


def reduce_frames(
        sim_frames: list[MBSPointCloud],
        ):
    """Reduces the target amount of frames from a simulation"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    simulate_structure("1ZNF", "1")
