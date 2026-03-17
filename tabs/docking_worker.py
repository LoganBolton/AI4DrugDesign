"""
Docking worker module for parallel compound docking.

This module is separate from integrated_pipeline.py to avoid importing Gradio
in worker processes, which would cause each worker to try launching a Gradio server.
"""

import os
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina


def smiles_to_pdbqt(smiles: str, output_path: str) -> None:
    """Convert SMILES to PDBQT file."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Exception(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        raise Exception("Failed to generate 3D coordinates")

    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    if not is_ok:
        raise Exception(f"PDBQT conversion failed: {error_msg}")

    with open(output_path, "w") as f:
        f.write(pdbqt_string)


def dock_single_compound(args: tuple) -> tuple[int, float]:
    """
    Worker function to dock a single compound (for parallel processing).

    Args:
        args: Tuple of (idx, smiles, receptor_pdbqt_path, center_coords)

    Returns:
        Tuple of (compound_index, vina_affinity_score)
    """
    idx, smiles, receptor_pdbqt, center = args

    try:
        # Create temporary directory for this compound
        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare ligand
            ligand_pdbqt = os.path.join(tmpdir, "ligand.pdbqt")
            smiles_to_pdbqt(smiles, ligand_pdbqt)

            # Run Vina (use 1 CPU per worker to avoid over-subscription)
            v = Vina(sf_name="vina", cpu=1, verbosity=0)
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_file(ligand_pdbqt)
            v.compute_vina_maps(center=list(center), box_size=[20, 20, 20])
            v.dock(exhaustiveness=4, n_poses=1)
            energies = v.energies(n_poses=1)

            affinity = float(energies[0][0])
            return (idx, affinity)

    except Exception as e:
        # Return inf to indicate failure
        return (idx, float('inf'))
