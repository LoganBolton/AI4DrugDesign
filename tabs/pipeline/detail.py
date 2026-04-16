"""Step 6 — Compound detail view, selection handler, AI explanation."""

import html as html_module
import io
import json

import gradio as gr
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

from tabs.pipeline.helpers import get_openai_client, logger


# ---------------------------------------------------------------------------
# Pharmacophore colour palette (RGB 0–1 tuples for rdMolDraw2D)
# ---------------------------------------------------------------------------
_DONOR_COLOR    = (0.25, 0.50, 1.00)   # blue  – H-bond donors  (N/O with H)
_ACCEPTOR_COLOR = (1.00, 0.30, 0.30)   # red   – H-bond acceptors (N/O lone-pair only)
_AROMATIC_COLOR = (0.90, 0.70, 0.10)   # gold  – aromatic atoms

# SMARTS patterns (matched on heavy-atom mol, no explicit Hs needed)
_SMARTS_DONOR    = Chem.MolFromSmarts('[N,O;!H0]')
_SMARTS_ACCEPTOR = Chem.MolFromSmarts('[N,O;H0]')
_SMARTS_AROMATIC = Chem.MolFromSmarts('[a]')


def _pharmacophore_highlights(mol):
    """Return (highlight_atoms, highlight_atom_colors) for rdMolDraw2D.

    Priority: donor > acceptor > aromatic.
    Highlights the binding-relevant atoms — those that interact directly
    with receptor residues via H-bonds or π-stacking.
    """
    seen = {}

    def _add(smarts, color):
        if smarts is None:
            return
        for match in mol.GetSubstructMatches(smarts):
            idx = match[0]
            if idx not in seen:
                seen[idx] = color

    _add(_SMARTS_DONOR,    _DONOR_COLOR)
    _add(_SMARTS_ACCEPTOR, _ACCEPTOR_COLOR)
    _add(_SMARTS_AROMATIC, _AROMATIC_COLOR)

    atoms  = list(seen.keys())
    colors = {idx: c for idx, c in seen.items()}
    return atoms, colors


def draw_2d_with_pharmacophore(mol, size=(400, 400)):
    """Render a 2D molecule image with pharmacophore-feature atom highlights.

    Returns a PIL Image. Falls back to plain MolToImage on any drawing error.
    """
    AllChem.Compute2DCoords(mol)
    highlight_atoms, highlight_colors = _pharmacophore_highlights(mol)

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        opts.addAtomIndices = False
        opts.addStereoAnnotation = True
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors,
        )
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText()))
    except Exception as exc:
        logger.warning(f"Pharmacophore 2D draw failed ({exc}), falling back to plain image")
        return Draw.MolToImage(mol, size=size)

COMPOUND_3D_PLACEHOLDER = (
    '<div style="height:400px;display:flex;align-items:center;'
    'justify-content:center;background:#f5f5f5;border-radius:8px;">'
    '<p style="color:#999;">Select a compound to view its 3D structure</p></div>'
)


def build_3d_compound_viewer_html(smiles: str, compound_name: str = "") -> str:
    """Build an iframe with 3Dmol.js to display a compound's 3D structure from SMILES.

    Converts SMILES → SDF via RDKit (3D coords + MMFF minimisation).  Atoms are
    coloured by pharmacophore role so binding-relevant features stand out:
      blue = H-bond donor   red = H-bond acceptor   gold = aromatic   grey = other (Jmol)
    """
    mol_orig = Chem.MolFromSmiles(smiles)
    if mol_orig is None:
        return (
            '<div style="height:400px;display:flex;align-items:center;'
            'justify-content:center;background:#f5f5f5;border-radius:8px;">'
            '<p style="color:#c00;">Could not parse SMILES for 3D generation</p></div>'
        )

    # Compute pharmacophore indices on the heavy-atom mol (indices stay valid
    # after AddHs because RDKit appends H atoms at the end)
    _, pharma_colors = _pharmacophore_highlights(mol_orig)

    # Group indices by colour for JS
    donors    = [i for i, c in pharma_colors.items() if c == _DONOR_COLOR]
    acceptors = [i for i, c in pharma_colors.items() if c == _ACCEPTOR_COLOR]
    aromatics = [i for i, c in pharma_colors.items() if c == _AROMATIC_COLOR]

    # Add hydrogens and generate 3D coordinates
    mol = Chem.AddHs(mol_orig)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)

    try:
        ff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        if ff_result != 0:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    sdf_block = Chem.MolToMolBlock(mol)
    sdf_json = json.dumps(sdf_block)

    # Pharmacophore index lists as JSON for JS
    donors_js    = json.dumps(donors)
    acceptors_js = json.dumps(acceptors)
    aromatics_js = json.dumps(aromatics)

    title_html = (
        f'<div style="padding:6px 10px;font-size:12px;color:#555;'
        f'border-bottom:1px solid #ddd;background:#fafafa;display:flex;'
        f'align-items:center;gap:12px;">'
        f'<b>3D Structure</b>'
        + (f': {html_module.escape(compound_name)}' if compound_name else '')
        + '<span style="margin-left:auto;font-size:11px;color:#777;">'
        '<span style="color:#4488ff">&#9679;</span> H-bond donor &nbsp;'
        '<span style="color:#ff4444">&#9679;</span> H-bond acceptor &nbsp;'
        '<span style="color:#cc9900">&#9679;</span> Aromatic'
        '</span>'
        '</div>'
    )

    inner = (
        "<!DOCTYPE html>"
        "<html><head>"
        "<script src='https://3Dmol.org/build/3Dmol-min.js'></script>"
        "<style>"
        "body{margin:0;padding:0;overflow:hidden;background:#fff}"
        "#viewer{width:100%;height:100%;position:absolute;top:0;left:0}"
        "</style>"
        "</head><body>"
        "<div id='viewer'></div>"
        "<script>"
        "window.addEventListener('load', function() {"
        "  var sdf = " + sdf_json + ";"
        "  var donors    = " + donors_js + ";"
        "  var acceptors = " + acceptors_js + ";"
        "  var aromatics = " + aromatics_js + ";"
        "  var viewer = $3Dmol.createViewer('viewer', {backgroundColor: 'white'});"
        "  viewer.addModel(sdf, 'sdf');"
        "  viewer.setStyle({}, {stick: {radius: 0.13, colorscheme: 'Jmol'}, sphere: {scale: 0.22, colorscheme: 'Jmol'}});"
        "  if (donors.length)    viewer.setStyle({index: donors},     {stick: {radius: 0.18, color: '#4488ff'}, sphere: {scale: 0.38, color: '#4488ff'}});"
        "  if (acceptors.length) viewer.setStyle({index: acceptors},  {stick: {radius: 0.18, color: '#ff4444'}, sphere: {scale: 0.38, color: '#ff4444'}});"
        "  if (aromatics.length) viewer.setStyle({index: aromatics},  {stick: {radius: 0.14, color: '#cc9900'}, sphere: {scale: 0.30, color: '#cc9900'}});"
        "  viewer.zoomTo();"
        "  viewer.render();"
        "});"
        "</script>"
        "</body></html>"
    )

    escaped = html_module.escape(inner, quote=True)
    return (
        f'<div style="border:1px solid #ddd;border-radius:8px;overflow:hidden;">'
        + title_html
        + f'<iframe srcdoc="{escaped}" '
        'style="width:100%;height:390px;border:none;display:block;"></iframe>'
        '</div>'
    )


def on_select_compound(evt: gr.SelectData, final_state):
    """Handle click on a row in the final results table."""
    if not final_state:
        logger.warning("Compound selection attempted with no compounds available")
        return "No compound selected.", None, COMPOUND_3D_PLACEHOLDER, None
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx >= len(final_state):
        logger.warning(f"Invalid compound index selected: {idx}")
        return "Invalid selection.", None, COMPOUND_3D_PLACEHOLDER, None

    compound = final_state[idx]
    smiles = compound["smiles"]
    name = compound.get("name", "")
    logger.info(f"Selected compound: {name} (ChEMBL: {compound.get('chembl_id')})")

    detail = (
        f"{'=' * 50}\n"
        f"  COMPOUND: {compound['name']}\n"
        f"{'=' * 50}\n"
        f"ChEMBL ID       : {compound.get('chembl_id', 'N/A')}\n"
        f"SMILES           : {smiles}\n\n"
        f"PROPERTIES:\n"
        f"  Molecular Weight : {compound.get('mw', 'N/A')}\n"
        f"  LogP             : {compound.get('logp', 'N/A')}\n"
        f"  H-Bond Donors    : {compound.get('hbd', 'N/A')}\n"
        f"  H-Bond Acceptors : {compound.get('hba', 'N/A')}\n"
        f"  TPSA             : {compound.get('tpsa', 'N/A')}\n"
        f"  Rotatable Bonds  : {compound.get('rot_bonds', 'N/A')}\n\n"
        f"BINDING DATA:\n"
        f"  Assay Type : {compound.get('activity_type', 'N/A')}\n"
        f"  Value      : {compound.get('activity_value', 'N/A')} "
        f"{compound.get('activity_units', '')}\n\n"
        f"FILTERS:\n"
        f"  Rule of 5  : "
        f"{'Pass' if compound.get('ro5_pass') else 'Fail'} "
        f"({compound.get('ro5_violations', '?')} violations)\n"
        f"  ADME Score : {compound.get('adme_score', 'N/A')}/6\n"
        f"{'=' * 50}"
    )

    mol = Chem.MolFromSmiles(smiles)
    img = None
    if mol:
        img = draw_2d_with_pharmacophore(mol, size=(400, 400))

    # Build 3D viewer HTML
    logger.info(f"Building 3D viewer for {name}")
    try:
        viewer_3d = build_3d_compound_viewer_html(smiles, name)
    except Exception as exc:
        logger.error(f"3D viewer generation failed for {name}: {exc}")
        viewer_3d = (
            '<div style="height:400px;display:flex;align-items:center;'
            'justify-content:center;background:#f5f5f5;border-radius:8px;">'
            f'<p style="color:#c00;">3D viewer error: {exc}</p></div>'
        )

    return detail, img, viewer_3d, compound


def ai_explain_compound(selected_state, protein_state):
    """AI-generated explanation of how the compound interacts with the protein."""
    logger.info("=== Generating AI compound explanation ===")
    if not selected_state:
        logger.warning("AI explanation requested with no compound selected")
        return "Select a compound from the results table first."
    if not protein_state:
        logger.warning("AI explanation requested with no protein data")
        return "No protein data available."

    client = get_openai_client()
    if not client:
        logger.warning("AI explanation requested but OpenAI API key not set")
        return "[OpenAI API key not set — cannot generate explanation]"

    c = selected_state
    p = protein_state
    logger.info(f"Generating AI explanation for {c.get('name')} targeting {p['pdb_id']}")

    prompt = (
        "You are an expert medicinal chemist. Explain how this compound "
        "works in the context of its protein target.\n\n"
        f"PROTEIN:\n"
        f"  Name: {p['title']}\n"
        f"  PDB ID: {p['pdb_id']}\n"
        f"  Classification: {p['classification']}\n"
        f"  Organism: {p['organism']}\n\n"
        f"COMPOUND:\n"
        f"  Name: {c.get('name', 'Unknown')}\n"
        f"  SMILES: {c.get('smiles', 'N/A')}\n"
        f"  MW: {c.get('mw', 'N/A')}, LogP: {c.get('logp', 'N/A')}\n"
        f"  Binding: {c.get('activity_type', 'N/A')} = "
        f"{c.get('activity_value', 'N/A')} {c.get('activity_units', '')}\n\n"
        "Explain:\n"
        "1. How this compound likely binds to the protein\n"
        "2. Key functional groups and their roles\n"
        "3. Mechanism of action (inhibitor, agonist, etc.)\n"
        "4. Strengths and weaknesses\n"
        "5. Potential optimization strategies\n\n"
        "Be concise and scientifically accurate."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": prompt}],
        )
        logger.info("AI compound explanation completed successfully")
        return resp.choices[0].message.content
    except Exception as exc:
        logger.error(f"AI compound explanation failed: {exc}")
        return f"AI analysis failed: {exc}"


def populate_compound_selector(final_state):
    """Build a summary table of final compounds for selection."""
    if not final_state:
        return None

    rows = []
    for c in final_state:
        vina = c.get("vina_affinity")
        rows.append({
            "Rank": c.get("binding_rank", ""),
            "Name": c["name"][:25],
            "Vina (kcal/mol)": (
                f"{vina:.2f}" if isinstance(vina, (int, float)) and vina != float("inf")
                else "N/A"
            ),
            "ADME": f"{c.get('adme_score', '?')}/6",
            "MW": c.get("mw", ""),
            "LogP": c.get("logp", ""),
        })

    logger.info(f"Populated compound selector with {len(rows)} compounds")
    return pd.DataFrame(rows)