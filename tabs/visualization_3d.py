"""
3D Visualization Tab
====================
Provides interactive 3D structure viewing for:
  - Small molecules  (SMILES → 3D conformer via RDKit, rendered with 3Dmol.js)
  - Proteins         (PDB ID  → fetched from RCSB, rendered with 3Dmol.js)
  - Protein + Ligand (PDB ID + SMILES → overlaid view)

3Dmol.js is loaded inside a sandboxed <iframe srcdoc=...> so that
Gradio's script-stripping does not block it.
"""

import re
import urllib.request
from urllib.error import HTTPError

import gradio as gr

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smiles_to_molblock(smiles: str) -> str:
    """Convert SMILES to a 3-D MOL-block string."""
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: '{smiles}'")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return Chem.MolToMolBlock(mol)


def _fetch_pdb_text(pdb_id: str) -> str:
    """Download a PDB file from RCSB as plain text."""
    pdb_id = pdb_id.strip().upper()
    if not re.match(r"^[A-Z0-9]{4}$", pdb_id):
        raise ValueError(f"Invalid PDB ID '{pdb_id}'. Must be 4 alphanumeric characters.")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except HTTPError as e:
        raise ValueError(f"PDB '{pdb_id}' not found (HTTP {e.code}).")
    except Exception as e:
        raise ValueError(f"Network error fetching '{pdb_id}': {e}")


def _mol_properties(smiles: str) -> str:
    """Return a short property summary string for a SMILES."""
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return ""
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    rot  = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    return (
        f"MW: {mw:.1f}  |  LogP: {logp:.2f}  |  "
        f"HBD: {hbd}  |  HBA: {hba}  |  "
        f"RotBonds: {rot}  |  TPSA: {tpsa:.1f} Å²"
    )


def _escape_for_js(text: str) -> str:
    """Escape a string so it can be safely embedded inside a JS template literal."""
    return (
        text
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("$", "\\$")
    )


def _wrap_in_iframe(inner_html: str) -> str:
    """
    Wrap HTML in an iframe using srcdoc so Gradio does not strip scripts.
    """
    escaped = inner_html.replace("&", "&amp;").replace('"', "&quot;")
    return f'<iframe srcdoc="{escaped}" style="width:100%;height:520px;border:none;border-radius:8px;"></iframe>'


# ---------------------------------------------------------------------------
# 3Dmol.js HTML builders
# ---------------------------------------------------------------------------

_CDN = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"


def _build_molecule_html(molblock: str, style: str, bg: str) -> str:
    mol_js = _escape_for_js(molblock)

    if style == "Surface":
        style_js = "viewer.setStyle({}, {stick:{}}); viewer.addSurface($3Dmol.SurfaceType.VWS, {opacity:0.75, colorscheme:'Jmol'});"
    elif style == "Stick+Surface":
        style_js = "viewer.setStyle({}, {stick:{}}); viewer.addSurface($3Dmol.SurfaceType.SAS, {opacity:0.5, colorscheme:'Jmol'});"
    elif style == "Sphere":
        style_js = "viewer.setStyle({}, {sphere:{scale:0.35}});"
    elif style == "Line":
        style_js = "viewer.setStyle({}, {line:{}});"
    else:  # Stick (default)
        style_js = "viewer.setStyle({}, {stick:{}});"

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="{_CDN}"></script>
  <style>
    html, body {{ margin:0; padding:0; width:100%; height:100%; background:{bg}; overflow:hidden; }}
    #v {{ width:100%; height:100%; position:relative; }}
  </style>
</head>
<body>
  <div id="v"></div>
  <script>
    window.addEventListener('load', function() {{
      var viewer = $3Dmol.createViewer(document.getElementById('v'), {{backgroundColor:'{bg}'}});
      viewer.addModel(`{mol_js}`, 'mol');
      {style_js}
      viewer.zoomTo();
      viewer.render();
    }});
  </script>
</body>
</html>"""


def _build_protein_html(pdb_text: str, style: str, color_scheme: str, bg: str) -> str:
    pdb_js = _escape_for_js(pdb_text)

    color_map = {
        "Spectrum":  "spectrum",
        "Chain":     "chain",
        "Secondary": "ssJmol",
        "B-factor":  "b",
        "Residue":   "residue",
    }
    color = color_map.get(color_scheme, "spectrum")

    if style == "Surface":
        style_js = f"""
          viewer.setStyle({{}}, {{cartoon:{{color:'{color}'}}}});
          viewer.addSurface($3Dmol.SurfaceType.SAS, {{opacity:0.65, colorscheme:'{color}'}});"""
    elif style == "Stick":
        style_js = "viewer.setStyle({}, {stick:{}});"
    elif style == "Sphere":
        style_js = "viewer.setStyle({}, {sphere:{scale:0.3}});"
    elif style == "Line":
        style_js = "viewer.setStyle({}, {line:{}});"
    else:  # Cartoon (default)
        style_js = f"viewer.setStyle({{}}, {{cartoon:{{color:'{color}'}}}});"

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="{_CDN}"></script>
  <style>
    html, body {{ margin:0; padding:0; width:100%; height:100%; background:{bg}; overflow:hidden; }}
    #v {{ width:100%; height:100%; position:relative; }}
  </style>
</head>
<body>
  <div id="v"></div>
  <script>
    window.addEventListener('load', function() {{
      var viewer = $3Dmol.createViewer(document.getElementById('v'), {{backgroundColor:'{bg}'}});
      viewer.addModel(`{pdb_js}`, 'pdb');
      {style_js}
      viewer.setStyle({{hetflag:true}}, {{stick:{{colorscheme:'yellowCarbon', radius:0.3}}}});
      viewer.zoomTo();
      viewer.render();
    }});
  </script>
</body>
</html>"""


def _build_complex_html(pdb_text: str, molblock: str, bg: str) -> str:
    pdb_js = _escape_for_js(pdb_text)
    mol_js = _escape_for_js(molblock)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="{_CDN}"></script>
  <style>
    html, body {{ margin:0; padding:0; width:100%; height:100%; background:{bg}; overflow:hidden; }}
    #v {{ width:100%; height:100%; position:relative; }}
  </style>
</head>
<body>
  <div id="v"></div>
  <script>
    window.addEventListener('load', function() {{
      var viewer = $3Dmol.createViewer(document.getElementById('v'), {{backgroundColor:'{bg}'}});
      viewer.addModel(`{pdb_js}`, 'pdb');
      viewer.setStyle({{model:0}}, {{cartoon:{{color:'spectrum', opacity:0.85}}}});
      viewer.setStyle({{model:0, hetflag:true}}, {{stick:{{colorscheme:'greenCarbon', radius:0.25}}}});
      viewer.addModel(`{mol_js}`, 'mol');
      viewer.setStyle({{model:1}}, {{stick:{{colorscheme:'cyanCarbon', radius:0.28}}}});
      viewer.addSurface($3Dmol.SurfaceType.VWS, {{opacity:0.55, colorscheme:'cyanCarbon'}}, {{model:1}});
      viewer.zoomTo();
      viewer.render();
    }});
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Gradio callback functions
# ---------------------------------------------------------------------------

def _visualize_molecule(smiles: str, style: str, bg: str):
    smiles = (smiles or "").strip()
    if not smiles:
        return "<p style='color:red;padding:1em'>Please enter a SMILES string.</p>", ""
    try:
        molblock = _smiles_to_molblock(smiles)
    except ValueError as e:
        return f"<p style='color:red;padding:1em'>{e}</p>", ""

    html = _build_molecule_html(molblock, style, bg)
    return _wrap_in_iframe(html), _mol_properties(smiles)


def _visualize_protein(pdb_id: str, style: str, color_scheme: str, bg: str):
    pdb_id = (pdb_id or "").strip().upper()
    if not pdb_id:
        return "<p style='color:red;padding:1em'>Please enter a PDB ID.</p>", ""
    try:
        pdb_text = _fetch_pdb_text(pdb_id)
    except ValueError as e:
        return f"<p style='color:red;padding:1em'>{e}</p>", ""

    atom_lines   = [l for l in pdb_text.splitlines() if l.startswith("ATOM")]
    hetatm_lines = [l for l in pdb_text.splitlines() if l.startswith("HETATM")]
    chains = sorted({l[21] for l in atom_lines if len(l) > 21})
    stats = (
        f"PDB: {pdb_id}  |  ATOM records: {len(atom_lines)}  |  "
        f"HETATM records: {len(hetatm_lines)}  |  Chains: {', '.join(chains) or 'N/A'}"
    )

    html = _build_protein_html(pdb_text, style, color_scheme, bg)
    return _wrap_in_iframe(html), stats


def _visualize_complex(pdb_id: str, smiles: str, bg: str):
    pdb_id = (pdb_id or "").strip().upper()
    smiles = (smiles or "").strip()
    errors = []
    if not pdb_id:
        errors.append("Please enter a PDB ID.")
    if not smiles:
        errors.append("Please enter a ligand SMILES.")
    if errors:
        return "<p style='color:red;padding:1em'>" + "<br>".join(errors) + "</p>", ""

    try:
        pdb_text = _fetch_pdb_text(pdb_id)
    except ValueError as e:
        return f"<p style='color:red;padding:1em'>{e}</p>", ""
    try:
        molblock = _smiles_to_molblock(smiles)
    except ValueError as e:
        return f"<p style='color:red;padding:1em'>{e}</p>", ""

    atom_lines = [l for l in pdb_text.splitlines() if l.startswith("ATOM")]
    chains = sorted({l[21] for l in atom_lines if len(l) > 21})
    info = (
        f"PDB: {pdb_id}  |  Chains: {', '.join(chains) or 'N/A'}  |  "
        f"Ligand — {_mol_properties(smiles)}"
    )

    html = _build_complex_html(pdb_text, molblock, bg)
    return _wrap_in_iframe(html), info


# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

def create_tab():
    with gr.Tab("3D Visualization"):
        gr.Markdown(
            "## Interactive 3D Molecular Viewer\n"
            "Visualize small molecules, proteins, or protein–ligand complexes in 3D.  \n"
            "**Drag** to rotate · **Scroll** to zoom · **Right-click drag** to translate."
        )

        with gr.Tabs():

            # ── Small Molecule ───────────────────────────────────────────────
            with gr.Tab("Small Molecule"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mol_smiles = gr.Textbox(
                            label="SMILES String",
                            placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O",
                            lines=3,
                        )
                        mol_style = gr.Radio(
                            choices=["Stick", "Sphere", "Line", "Surface", "Stick+Surface"],
                            value="Stick",
                            label="Display Style",
                        )
                        mol_bg  = gr.ColorPicker(value="#1a1a2e", label="Background Color")
                        mol_btn = gr.Button("Visualize Molecule", variant="primary", size="lg")
                        gr.Examples(
                            examples=[
                                ["CC(=O)Oc1ccccc1C(=O)O"],
                                ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
                                ["CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"],
                                ["CC(C)Cc1ccc(C(C)C(=O)O)cc1"],
                                ["c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34"],
                            ],
                            inputs=[mol_smiles],
                            label="Examples (Aspirin, Caffeine, Testosterone, Ibuprofen, Pyrene)",
                        )
                    with gr.Column(scale=2):
                        mol_viewer = gr.HTML(label="3D Viewer")
                        mol_props  = gr.Textbox(label="Molecular Properties", interactive=False, lines=2)

                mol_btn.click(
                    _visualize_molecule,
                    inputs=[mol_smiles, mol_style, mol_bg],
                    outputs=[mol_viewer, mol_props],
                )

            # ── Protein Structure ────────────────────────────────────────────
            with gr.Tab("Protein Structure"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prot_pdb   = gr.Textbox(label="PDB ID", placeholder="e.g., 6LU7", lines=1)
                        prot_style = gr.Radio(
                            choices=["Cartoon", "Surface", "Stick", "Sphere", "Line"],
                            value="Cartoon",
                            label="Display Style",
                        )
                        prot_color = gr.Radio(
                            choices=["Spectrum", "Chain", "Secondary", "B-factor", "Residue"],
                            value="Spectrum",
                            label="Color Scheme",
                        )
                        prot_bg  = gr.ColorPicker(value="#0d0d1a", label="Background Color")
                        prot_btn = gr.Button("Visualize Protein", variant="primary", size="lg")
                        gr.Examples(
                            examples=[["6LU7"], ["1IEP"], ["1AZ5"], ["2HYY"], ["1HSG"]],
                            inputs=[prot_pdb],
                            label="Examples (SARS-CoV-2, Abl kinase, ER, CDK2, HIV protease)",
                        )
                    with gr.Column(scale=2):
                        prot_viewer = gr.HTML(label="3D Viewer")
                        prot_stats  = gr.Textbox(label="Structure Info", interactive=False, lines=2)

                prot_btn.click(
                    _visualize_protein,
                    inputs=[prot_pdb, prot_style, prot_color, prot_bg],
                    outputs=[prot_viewer, prot_stats],
                )

            # ── Protein–Ligand Complex ───────────────────────────────────────
            with gr.Tab("Protein–Ligand Complex"):
                gr.Markdown(
                    "Co-crystallized ligands are shown in **green**. "
                    "Your input ligand (from SMILES) is shown in **cyan**."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        cplx_pdb    = gr.Textbox(label="PDB ID", placeholder="e.g., 1IEP", lines=1)
                        cplx_smiles = gr.Textbox(
                            label="Ligand SMILES",
                            placeholder="e.g., CC(C)Cc1ccc(C(C)C(=O)O)cc1",
                            lines=3,
                        )
                        cplx_bg  = gr.ColorPicker(value="#0a0a1a", label="Background Color")
                        cplx_btn = gr.Button("Visualize Complex", variant="primary", size="lg")
                        gr.Examples(
                            examples=[
                                ["1IEP", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"],
                                ["6LU7", "CC(C)[C@@H](NC(=O)[C@@H]1C[C@H]1c1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C=O"],
                                ["1HSG", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"],
                            ],
                            inputs=[cplx_pdb, cplx_smiles],
                            label="Examples",
                        )
                    with gr.Column(scale=2):
                        cplx_viewer = gr.HTML(label="3D Viewer")
                        cplx_info   = gr.Textbox(label="Complex Info", interactive=False, lines=2)

                cplx_btn.click(
                    _visualize_complex,
                    inputs=[cplx_pdb, cplx_smiles, cplx_bg],
                    outputs=[cplx_viewer, cplx_info],
                )