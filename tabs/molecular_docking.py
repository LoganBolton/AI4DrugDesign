import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from urllib.error import HTTPError

import gradio as gr
import numpy as np
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem
#from vina import Vina


# Common non-drug HETATM residues to ignore when searching for a co-ligand
_EXCLUDED_HETATM = frozenset(
    {
        "HOH",
        "WAT",
        "SO4",
        "GOL",
        "EDO",
        "PEG",
        "PGE",
        "MPD",
        "DMS",
        "ACT",
        "BME",
        "CL",
        "NA",
        "MG",
        "ZN",
        "CA",
        "K",
        "MN",
        "FE",
        "CU",
        "CO",
        "NI",
        "CD",
        "IOD",
        "BR",
        "FMT",
        "NO3",
        "PO4",
        "NH4",
        "CSO",
        "OCS",
        "CME",
        "TRS",
        "EPE",
        "IMD",
        "MES",
        "CIT",
        "HED",
        "IPA",
        "BU3",
        "1PE",
        "2PE",
        "P6G",
        "SIN",
    }
)


def _fetch_pdb(pdb_id: str) -> str:
    """Fetch PDB file text from RCSB."""
    pdb_id = pdb_id.strip().upper()
    if not re.match(r"^[A-Z0-9]{4}$", pdb_id):
        raise gr.Error(f"Invalid PDB ID: '{pdb_id}'. Must be exactly 4 alphanumeric characters.")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except HTTPError as e:
        if e.code == 404:
            raise gr.Error(f"PDB ID '{pdb_id}' not found on RCSB.")
        raise gr.Error(f"Failed to fetch PDB '{pdb_id}': HTTP {e.code}")
    except Exception as e:
        raise gr.Error(f"Network error fetching PDB '{pdb_id}': {e}")


def _find_coligand_center(pdb_text: str) -> tuple[float, float, float]:
    """Parse HETATM records to find the largest non-solvent ligand and return its geometric center."""
    residue_coords: dict[str, list[tuple[float, float, float]]] = {}

    for line in pdb_text.splitlines():
        if not line.startswith("HETATM"):
            continue
        res_name = line[17:20].strip()
        if res_name in _EXCLUDED_HETATM:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue
        residue_coords.setdefault(res_name, []).append((x, y, z))

    if not residue_coords:
        raise gr.Error("No co-crystallized ligand found in this PDB structure.")

    # Pick the residue with the most atoms
    best_res = max(residue_coords, key=lambda r: len(residue_coords[r]))
    coords = np.array(residue_coords[best_res])
    center = coords.mean(axis=0)
    return round(float(center[0]), 3), round(float(center[1]), 3), round(float(center[2]), 3)


def _smiles_to_pdbqt(smiles: str, output_path: str) -> None:
    """Convert a SMILES string to a PDBQT file using RDKit + meeko."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise gr.Error(f"Invalid SMILES string: '{smiles}'")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        raise gr.Error("Failed to generate 3D coordinates for this SMILES. Try a simpler molecule.")

    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    if not is_ok:
        raise gr.Error(f"Meeko PDBQT conversion failed: {error_msg}")

    with open(output_path, "w") as f:
        f.write(pdbqt_string)


def _clean_pdb_for_receptor(pdb_path: str, cleaned_path: str) -> None:
    """Strip HETATM records and keep only standard protein atoms for receptor preparation."""
    with open(pdb_path) as f:
        lines = f.readlines()
    with open(cleaned_path, "w") as f:
        for line in lines:
            if line.startswith(("ATOM", "TER", "END", "MODEL", "ENDMDL", "REMARK", "CRYST1")):
                f.write(line)


def _prepare_receptor_pdbqt(pdb_path: str, output_path: str) -> None:
    """Prepare receptor PDBQT from a PDB file using mk_prepare_receptor.py (meeko)."""
    mk_script = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")
    if mk_script is None:
        raise gr.Error(
            "mk_prepare_receptor not found. It should be installed with meeko. "
            "Try: pip install meeko"
        )

    # Clean PDB to remove HETATM records (ligands, cofactors, waters) that
    # cause meeko to fail on unknown residues.
    cleaned_path = pdb_path.replace(".pdb", "_clean.pdb")
    _clean_pdb_for_receptor(pdb_path, cleaned_path)

    # -o sets the output basename (meeko appends .pdbqt), -p writes PDBQT
    output_basename = output_path.removesuffix(".pdbqt")
    cmd = [mk_script, "-i", cleaned_path, "-o", output_basename, "-p"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise gr.Error("Receptor preparation timed out after 120 seconds.")

    if result.returncode != 0:
        stderr_msg = result.stderr[:500] if result.stderr else "No error output"
        raise gr.Error(f"Receptor preparation failed:\n{stderr_msg}")


def _format_vina_energies(energies: np.ndarray) -> str:
    """Format Vina energy array into a readable affinity table.

    energies columns: [affinity, inter, intra, torsion, unbound]
    """
    header = "mode |  affinity  |  inter   |  intra   | torsion"
    subhdr = "     | (kcal/mol) |          |          |        "
    sep    = "-----+------------+----------+----------+--------"
    lines = [header, subhdr, sep]
    for i, row in enumerate(energies, start=1):
        affinity = row[0]
        inter = row[1] if len(row) > 1 else 0.0
        intra = row[2] if len(row) > 2 else 0.0
        torsion = row[3] if len(row) > 3 else 0.0
        lines.append(f"  {i:>2} | {affinity:>10.3f} | {inter:>8.3f} | {intra:>8.3f} | {torsion:>6.3f}")
    return "\n".join(lines)


def _auto_calculate_center(pdb_id, smiles):
    """Fetch PDB structure and compute the geometric center of the co-crystallized ligand."""
    if not pdb_id or not pdb_id.strip():
        raise gr.Error("Please enter a PDB ID first.")

    pdb_text = _fetch_pdb(pdb_id)
    x, y, z = _find_coligand_center(pdb_text)
    gr.Info(f"Binding box center calculated from co-ligand: ({x}, {y}, {z})")
    return x, y, z


def _run_docking(pdb_id, smiles, center_x, center_y, center_z):
    """Full docking pipeline: prepare receptor & ligand, run Vina, return results."""
    # Validate inputs
    if not pdb_id or not pdb_id.strip():
        raise gr.Error("Please enter a PDB ID.")
    if not smiles or not smiles.strip():
        raise gr.Error("Please enter a ligand SMILES string.")

    pdb_text = _fetch_pdb(pdb_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write PDB file
        pdb_path = os.path.join(tmpdir, "receptor.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_text)

        # Prepare receptor PDBQT
        receptor_pdbqt = os.path.join(tmpdir, "receptor.pdbqt")
        _prepare_receptor_pdbqt(pdb_path, receptor_pdbqt)

        # Prepare ligand PDBQT
        ligand_pdbqt = os.path.join(tmpdir, "ligand.pdbqt")
        _smiles_to_pdbqt(smiles.strip(), ligand_pdbqt)

        # Run Vina via Python API
        output_pdbqt = os.path.join(tmpdir, "docked_output.pdbqt")
        try:
            v = Vina(sf_name="vina")
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_file(ligand_pdbqt)
            v.compute_vina_maps(
                center=[float(center_x), float(center_y), float(center_z)],
                box_size=[25, 25, 25],
            )
            v.dock(exhaustiveness=8, n_poses=9)
            v.write_poses(output_pdbqt, n_poses=9, overwrite=True)
            energies = v.energies(n_poses=9)
        except Exception as e:
            raise gr.Error(f"Vina docking failed:\n{e}")

        # Format affinities
        affinities = _format_vina_energies(energies)

        # Copy docked output to a persistent temp file for Gradio download
        if os.path.exists(output_pdbqt):
            persistent_output = tempfile.NamedTemporaryFile(
                delete=False, suffix="_docked.pdbqt", prefix="vina_"
            )
            persistent_output.close()
            shutil.copy2(output_pdbqt, persistent_output.name)
            return affinities, persistent_output.name
        else:
            return affinities, None


def create_tab():
    with gr.Tab("Molecular Docking"):
        with gr.Row():
            with gr.Column(scale=1):
                vina_pdb_input = gr.Textbox(
                    label="Target PDB ID",
                    placeholder="e.g., 6LU7",
                )
                vina_smiles_input = gr.Textbox(
                    label="Ligand SMILES",
                    placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
                    lines=3,
                )
                auto_calc_btn = gr.Button(
                    "Auto-Calculate Center from Co-Ligand",
                    size="sm",
                    elem_classes=["analyze-btn"],
                )
                with gr.Row():
                    center_x = gr.Number(label="Center X", value=0.0, precision=3)
                    center_y = gr.Number(label="Center Y", value=0.0, precision=3)
                    center_z = gr.Number(label="Center Z", value=0.0, precision=3)
                run_vina_btn = gr.Button(
                    "Run AutoDock Vina",
                    size="lg",
                    elem_classes=["analyze-btn"],
                )

            with gr.Column(scale=1):
                vina_affinities = gr.Textbox(
                    label="Vina Binding Affinities",
                    interactive=False,
                    lines=8,
                )
                vina_output_file = gr.File(
                    label="Docked PDBQT File (Download)",
                )

        gr.Examples(
            examples=[
                ["6LU7", "CC(=O)OC1=CC=CC=C1C(=O)O"],
                ["1IEP", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"],
            ],
            inputs=[vina_pdb_input, vina_smiles_input],
        )

        auto_calc_btn.click(
            _auto_calculate_center,
            inputs=[vina_pdb_input, vina_smiles_input],
            outputs=[center_x, center_y, center_z],
        )
        run_vina_btn.click(
            _run_docking,
            inputs=[vina_pdb_input, vina_smiles_input, center_x, center_y, center_z],
            outputs=[vina_affinities, vina_output_file],
        )
