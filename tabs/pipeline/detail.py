"""Step 6 — Compound detail view, selection handler, AI explanation."""

import html as html_module
import json

import gradio as gr
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from tabs.pipeline.helpers import get_openai_client, logger

COMPOUND_3D_PLACEHOLDER = (
    '<div style="height:400px;display:flex;align-items:center;'
    'justify-content:center;background:#f5f5f5;border-radius:8px;">'
    '<p style="color:#999;">Select a compound to view its 3D structure</p></div>'
)

DOCKED_3D_PLACEHOLDER = (
    '<div style="height:520px;display:flex;align-items:center;'
    'justify-content:center;background:#f0f4ff;border-radius:8px;border:1px solid #c8d8f0;">'
    '<p style="color:#666;text-align:center;padding:20px;">'
    'Select a compound, then click<br><b>Show Compound in Binding Site</b>'
    '<br><br><span style="font-size:11px;color:#999;">'
    'Renders the protein with the compound positioned at the binding pocket</span>'
    '</p></div>'
)


def build_3d_compound_viewer_html(smiles: str, compound_name: str = "") -> str:
    """Build an iframe with 3Dmol.js to display a compound's 3D structure from SMILES.

    Converts SMILES → SDF via RDKit (with 3D coordinates + MMFF force-field
    minimisation), then passes the SDF string directly into the 3Dmol viewer so
    no external fetch is needed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (
            '<div style="height:400px;display:flex;align-items:center;'
            'justify-content:center;background:#f5f5f5;border-radius:8px;">'
            '<p style="color:#c00;">Could not parse SMILES for 3D generation</p></div>'
        )

    # Add hydrogens and generate 3D coordinates
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        # Fallback: use distance-geometry with random coords
        AllChem.EmbedMolecule(mol, randomSeed=42)

    # Energy minimise with MMFF if available, else UFF
    try:
        ff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        if ff_result != 0:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    sdf_block = Chem.MolToMolBlock(mol)

    # Escape the SDF for safe embedding inside a JS string literal
    # We use JSON encoding so newlines/quotes are handled correctly
    import json
    sdf_json = json.dumps(sdf_block)

    title_html = (
        f'<div style="padding:6px 10px;font-size:12px;color:#555;'
        f'border-bottom:1px solid #ddd;background:#fafafa;">'
        f'<b>3D Structure</b>'
        + (f': {html_module.escape(compound_name)}' if compound_name else '')
        + '</div>'
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
        "  var config = {backgroundColor: 'white'};"
        "  var viewer = $3Dmol.createViewer('viewer', config);"
        "  viewer.addModel(sdf, 'sdf');"
        "  viewer.setStyle({}, {"
        "    stick: {radius: 0.15, colorscheme: 'Jmol'},"
        "    sphere: {scale: 0.25, colorscheme: 'Jmol'}"
        "  });"
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
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(400, 400))

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


def build_protein_ligand_viewer_html(
    smiles: str, pdb_id: str, compound_name: str = "", binding_center=None,
    pdb_text: str = "",
) -> str:
    """Build a 3Dmol.js iframe showing the protein + compound at the binding site.

    The compound's 3D centroid is translated to the detected binding-site centre so
    it is placed inside the pocket for visual inspection (not a true docked pose).

    If *pdb_text* is supplied it is embedded directly in the HTML so the browser
    does not need to fetch the PDB from RCSB again (avoids a redundant download).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (
            '<div style="height:520px;display:flex;align-items:center;'
            'justify-content:center;background:#f5f5f5;border-radius:8px;">'
            '<p style="color:#c00;">Could not parse SMILES for 3D generation</p></div>'
        )

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
    try:
        ff = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        if ff != 0:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    # Translate compound centroid to binding site centre
    cx, cy, cz = (binding_center if binding_center else (0.0, 0.0, 0.0))
    try:
        conf = mol.GetConformer()
        positions = conf.GetPositions()          # numpy array (N, 3)
        centroid = positions.mean(axis=0)
        dx, dy, dz = cx - centroid[0], cy - centroid[1], cz - centroid[2]
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (pos.x + dx, pos.y + dy, pos.z + dz))
    except Exception as exc:
        logger.warning(f"Could not translate compound to binding site: {exc}")

    sdf_block = Chem.MolToMolBlock(mol)
    sdf_json = json.dumps(sdf_block)

    pdb_id_safe = pdb_id.strip().upper()
    title_label = (
        f"Compound in Binding Site: {html_module.escape(compound_name)}"
        if compound_name
        else "Compound in Binding Site"
    )

    title_html = (
        f'<div style="padding:6px 12px;font-size:12px;color:#444;'
        f'border-bottom:1px solid #ddd;background:#f0f4ff;'
        f'display:flex;justify-content:space-between;align-items:center;">'
        f'<b>{title_label}</b>'
        f'<span style="color:#888;font-size:11px;">'
        f'Protein: semi-transparent cartoon &nbsp;|&nbsp; '
        f'Compound: stick/sphere &nbsp;|&nbsp; '
        f'Yellow sphere: binding region</span>'
        f'</div>'
    )

    # JS that runs once pdbData is in scope (viewer/sdf/cx/cy/cz are closures)
    render_js = (
        "  document.getElementById('loading').style.display = 'none';"
        "  viewer.addModel(pdbData, 'pdb');"
        "  var n = viewer.getNumModels();"
        "  viewer.setStyle({}, {cartoon: {color: 'spectrum', opacity: 0.72}});"
        "  viewer.addModel(sdf, 'sdf');"
        "  viewer.setStyle({model: n}, {"
        "    stick: {radius: 0.18, colorscheme: 'Jmol'},"
        "    sphere: {scale: 0.28, colorscheme: 'Jmol'}"
        "  });"
        "  if (cx !== 0 || cy !== 0 || cz !== 0) {"
        "    viewer.addSphere({center: {x: cx, y: cy, z: cz},"
        "      radius: 9, color: 'yellow', opacity: 0.09, wireframe: false});"
        "    viewer.addSphere({center: {x: cx, y: cy, z: cz},"
        "      radius: 9.2, color: 'orange', opacity: 0.18, wireframe: true});"
        "  }"
        "  viewer.zoomTo({model: n});"
        "  viewer.render();"
    )

    if pdb_text:
        pdb_json = json.dumps(pdb_text)
        scene_js = (
            "  var pdbData = " + pdb_json + ";"
            + render_js
        )
    else:
        scene_js = (
            f"  fetch('https://files.rcsb.org/download/{pdb_id_safe}.pdb')"
            "    .then(function(r){return r.ok?r.text():Promise.reject(r.status);})"
            "    .then(function(pdbData){"
            + render_js +
            "    })"
            "    .catch(function(){"
            "      document.getElementById('loading').textContent='Failed to load protein structure.';"
            "    });"
        )

    inner = (
        "<!DOCTYPE html><html><head>"
        "<script src='https://3Dmol.org/build/3Dmol-min.js'></script>"
        "<style>"
        "body{margin:0;padding:0;overflow:hidden;background:#fff;font-family:sans-serif}"
        "#viewer{width:100%;height:100%;position:absolute;top:0;left:0}"
        "#loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);"
        "color:#555;font-size:13px;z-index:10;pointer-events:none;"
        "background:rgba(255,255,255,0.85);padding:8px 16px;border-radius:6px;}"
        "</style>"
        "</head><body>"
        "<div id='viewer'></div>"
        "<div id='loading'>Loading protein structure&hellip;</div>"
        "<script>"
        "window.addEventListener('load', function() {"
        "  var sdf = " + sdf_json + ";"
        f"  var cx = {cx}, cy = {cy}, cz = {cz};"
        "  var viewer = $3Dmol.createViewer('viewer', {backgroundColor: 'white'});"
        + scene_js +
        "});"
        "</script>"
        "</body></html>"
    )

    escaped = html_module.escape(inner, quote=True)
    return (
        '<div style="border:1px solid #c8d8f0;border-radius:8px;overflow:hidden;">'
        + title_html
        + f'<iframe srcdoc="{escaped}" '
        'style="width:100%;height:510px;border:none;display:block;"></iframe>'
        '</div>'
    )


def show_docked_view(selected_state, protein_state):
    """Callback: render the selected compound positioned at the protein binding site."""
    if not selected_state or not protein_state:
        return DOCKED_3D_PLACEHOLDER

    smiles = selected_state.get("smiles", "")
    name = selected_state.get("name", "")
    pdb_id = protein_state.get("pdb_id", "")

    if not smiles or not pdb_id:
        return DOCKED_3D_PLACEHOLDER

    # Detect binding centre from the crystal structure.
    # Keep the fetched PDB text so we can embed it directly in the HTML
    # instead of downloading it a second time inside the browser iframe.
    binding_center = None
    pdb_text = ""
    try:
        from tabs.pipeline.docking import fetch_pdb_file, find_binding_center
        pdb_text = fetch_pdb_file(pdb_id)
        binding_center = find_binding_center(pdb_text)
        logger.info(f"Binding centre for {pdb_id}: {binding_center}")
    except Exception as exc:
        logger.warning(f"Could not determine binding centre: {exc}")

    try:
        return build_protein_ligand_viewer_html(smiles, pdb_id, name, binding_center, pdb_text)
    except Exception as exc:
        logger.error(f"Docked view generation failed: {exc}")
        return (
            '<div style="height:520px;display:flex;align-items:center;'
            'justify-content:center;background:#f5f5f5;border-radius:8px;">'
            f'<p style="color:#c00;">Viewer error: {html_module.escape(str(exc))}</p></div>'
        )


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