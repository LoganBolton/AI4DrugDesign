"""
Integrated Drug Discovery Pipeline

Single-page workflow:
  1. Input Protein (PDB ID) → Analyze & 3D view
  2. Discover Compounds (ChEMBL)
  3. Filter: Rule of 5
  4. Rank: Binding Activity (experimental data)
  5. Filter: ADME scores
  6. Select compound → Detail view + AI explanation
"""

import html as html_module
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.error import HTTPError

import gradio as gr
import numpy as np
import pandas as pd
import requests
from meeko import MoleculePreparation, PDBQTWriterLegacy
from openai import OpenAI
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw

# Import docking worker from separate module (avoids Gradio in worker processes)
from tabs.docking_worker import dock_single_compound

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drug_discovery_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _build_3d_viewer_html(pdb_id: str) -> str:
    """Build an iframe with 3Dmol.js to display a protein structure."""
    inner = (
        "<!DOCTYPE html>"
        "<html><head>"
        "<script src='https://3Dmol.org/build/3Dmol-min.js'></script>"
        "<style>body{margin:0;padding:0;overflow:hidden}"
        "#v{width:100%;height:100%;position:absolute}</style>"
        "</head><body><div id='v'></div><script>"
        "$(function(){"
        "var v=$3Dmol.createViewer('v',{backgroundColor:'white'});"
        "jQuery.ajax({url:'https://files.rcsb.org/download/" + pdb_id + ".pdb',"
        "success:function(d){"
        "v.addModel(d,'pdb');"
        "v.setStyle({},{cartoon:{color:'spectrum'}});"
        "v.zoomTo();v.render();},"
        "error:function(){"
        "document.getElementById('v').innerHTML="
        "'<p style=padding:20px>Could not load structure.</p>';}});"
        "});</script></body></html>"
    )
    escaped = html_module.escape(inner, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        'style="width:100%;height:450px;border:none;border-radius:8px;"></iframe>'
    )


_VIEWER_PLACEHOLDER = (
    '<div style="height:450px;display:flex;align-items:center;'
    "justify-content:center;background:#f5f5f5;border-radius:8px;\">"
    "<p style=\"color:#999;\">Enter a PDB ID and click "
    "'Analyze Protein' to view 3D structure</p></div>"
)


# ── Step 1 — Protein Analysis ─────────────────────────────────────────

_SKIP_LIGANDS = frozenset({
    "HOH", "SO4", "PO4", "GOL", "EDO", "ACT", "FMT", "IOD",
    "CL", "MG", "ZN", "CA", "NA", "PEG", "DMS",
})


def _fetch_protein_info(pdb_id: str) -> dict:
    """Fetch protein metadata from RCSB PDB."""
    pdb_id = pdb_id.strip().upper()
    logger.info(f"Fetching protein info for PDB ID: {pdb_id}")
    base = "https://data.rcsb.org/rest/v1/core"

    r = requests.get(f"{base}/entry/{pdb_id}", timeout=15)
    r.raise_for_status()
    entry = r.json()
    logger.debug(f"Retrieved entry data for {pdb_id}")

    title = entry.get("struct", {}).get("title", "N/A")
    keywords = entry.get("struct_keywords", {}).get("pdbx_keywords", "N/A")

    # Organism
    organisms = []
    try:
        pr = requests.get(f"{base}/entry/{pdb_id}/polymer_entities", timeout=15)
        if pr.ok:
            for ent in pr.json():
                for s in ent.get("rcsb_entity_source_organism", []):
                    org = s.get("ncbi_scientific_name")
                    if org and org not in organisms:
                        organisms.append(org)
        logger.debug(f"Found {len(organisms)} organism(s) for {pdb_id}")
    except Exception as e:
        logger.warning(f"Failed to fetch organisms for {pdb_id}: {e}")

    # Resolution
    resolution = (
        entry.get("refine", [{}])[0].get("ls_d_res_high")
        if entry.get("refine")
        else None
    )

    # Ligands
    non_poly_ids = (
        entry.get("rcsb_entry_container_identifiers", {})
        .get("non_polymer_entity_ids") or []
    )
    ligands = []
    for eid in non_poly_ids[:10]:
        try:
            er = requests.get(
                f"{base}/nonpolymer_entity/{pdb_id}/{eid}", timeout=10
            )
            if er.ok:
                d = er.json().get("pdbx_entity_nonpoly", {})
                comp_id = d.get("comp_id", "")
                name = d.get("name", comp_id)
                if comp_id and comp_id not in _SKIP_LIGANDS:
                    ligands.append({"comp_id": comp_id, "name": name})
        except Exception as e:
            logger.debug(f"Error fetching ligand entity {eid}: {e}")
            continue

    logger.info(f"Found {len(ligands)} ligand(s) in {pdb_id}")

    # Binding sites
    binding_sites = []
    try:
        sr = requests.get(
            f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/binding_sites/{pdb_id.lower()}",
            timeout=15,
        )
        if sr.ok:
            for sid, sinfo in sr.json().get(pdb_id.lower(), {}).items():
                det = sinfo[0] if isinstance(sinfo, list) and sinfo else sinfo
                binding_sites.append({
                    "id": sid,
                    "description": det.get("site_description", sid),
                })
        logger.info(f"Found {len(binding_sites)} binding site(s) in {pdb_id}")
    except Exception as e:
        logger.warning(f"Failed to fetch binding sites for {pdb_id}: {e}")

    logger.info(f"Successfully fetched complete protein info for {pdb_id}")
    return {
        "pdb_id": pdb_id,
        "title": title,
        "classification": keywords,
        "organism": ", ".join(organisms) if organisms else "N/A",
        "resolution": f"{resolution} \u00c5" if resolution else "N/A",
        "ligands": ligands,
        "binding_sites": binding_sites,
    }


def _get_uniprot_from_pdb(pdb_id: str) -> str | None:
    """Map PDB ID to UniProt accession via SIFTS."""
    try:
        r = requests.get(
            f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}",
            timeout=15,
        )
        if r.ok:
            data = r.json().get(pdb_id.lower(), {}).get("UniProt", {})
            if data:
                return list(data.keys())[0]
    except Exception:
        pass
    return None


def _update_protein_status():
    """Show status when protein analysis starts."""
    return "**Protein:** 🔄 Analyzing structure from RCSB PDB..."


def _run_protein_and_compounds_parallel(pdb_id):
    """Run protein analysis and compound discovery in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    logger.info("=== Starting parallel execution: Protein Analysis + Compound Discovery ===")

    # Initialize return values
    protein_results = None
    compound_results = None

    # Run both tasks in parallel using threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        protein_future = executor.submit(_analyze_protein, pdb_id)
        compound_future = executor.submit(_fetch_compounds, pdb_id)

        # Wait for both to complete (they run in parallel)
        protein_results = protein_future.result()
        compound_results = compound_future.result()

    logger.info("=== Parallel execution complete ===")

    # Return all results
    # protein_results: (status, text, info, viewer)
    # compound_results: (status, compounds, table)
    return (
        protein_results[0],  # protein_status
        protein_results[1],  # protein_text
        protein_results[2],  # protein_state
        protein_results[3],  # viewer_html
        compound_results[0],  # compounds_status
        compound_results[1],  # all_compounds_state
        compound_results[2],  # compounds_table
    )


def _run_complete_pipeline(pdb_id, num_dock=20):
    """Run the entire drug discovery pipeline from start to finish."""
    logger.info("=" * 70)
    logger.info("=== STARTING COMPLETE DRUG DISCOVERY PIPELINE ===")
    logger.info(f"=== PDB ID: {pdb_id}, Docking: {num_dock} compounds ===")
    logger.info("=" * 70)

    # Step 1 & 2: Protein analysis + Compound discovery (parallel)
    logger.info("PIPELINE: Step 1 & 2 - Protein Analysis + Compound Discovery")
    (protein_status, protein_text, protein_state, viewer_html,
     compounds_status, all_compounds_state, compounds_table) = \
        _run_protein_and_compounds_parallel(pdb_id)

    # Step 3: Rule of 5 filter
    logger.info("PIPELINE: Step 3 - Rule of 5 Filter")
    ro5_status, ro5_state, ro5_table = _apply_rule_of_5(all_compounds_state)

    # Step 4: AutoDock Vina docking
    logger.info("PIPELINE: Step 4 - AutoDock Vina Docking")
    rank_status, ranked_state, rank_table = _rank_by_activity(
        ro5_state, protein_state, num_dock
    )

    # Step 5: ADME filter
    logger.info("PIPELINE: Step 5 - ADME Filter")
    adme_status, final_state, adme_table = _apply_adme_filter(ranked_state)

    # Step 6: Populate compound selector
    logger.info("PIPELINE: Step 6 - Populating Compound Selector")
    compound_selector_update = _populate_compound_selector(final_state)

    logger.info("=" * 70)
    logger.info("=== COMPLETE PIPELINE FINISHED ===")
    logger.info(f"=== Final compounds: {len(final_state) if final_state else 0} ===")
    logger.info("=" * 70)

    # Return all outputs for UI update
    return (
        protein_status,
        protein_text,
        protein_state,
        viewer_html,
        compounds_status,
        all_compounds_state,
        compounds_table,
        ro5_status,
        ro5_state,
        ro5_table,
        rank_status,
        ranked_state,
        rank_table,
        adme_status,
        final_state,
        adme_table,
        compound_selector_update,
    )


def _analyze_protein(pdb_id: str):
    """Step 1 handler: fetch info, build display text, 3D viewer."""
    logger.info(f"=== STEP 1: Analyzing protein {pdb_id} ===")
    pdb_id = (pdb_id or "").strip().upper()
    if not pdb_id:
        logger.warning("Empty PDB ID provided")
        return (
            "**Protein:** ❌ Please enter a PDB ID",
            "Please enter a PDB ID.",
            None,
            _VIEWER_PLACEHOLDER,
        )

    try:
        info = _fetch_protein_info(pdb_id)
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        logger.error(f"HTTP error fetching {pdb_id}: {code}")
        if code == 404:
            return (
                f"**Protein:** ❌ PDB ID '{pdb_id}' not found",
                f"PDB ID '{pdb_id}' not found.",
                None,
                _VIEWER_PLACEHOLDER,
            )
        return (
            f"**Protein:** ❌ Error fetching data (HTTP {code})",
            f"Error fetching PDB data (HTTP {code}).",
            None,
            _VIEWER_PLACEHOLDER,
        )
    except Exception as exc:
        logger.error(f"Error analyzing protein {pdb_id}: {exc}")
        return (
            f"**Protein:** ❌ Error: {exc}",
            f"Error: {exc}",
            None,
            _VIEWER_PLACEHOLDER,
        )

    info["uniprot_id"] = _get_uniprot_from_pdb(pdb_id)
    logger.info(f"UniProt ID for {pdb_id}: {info['uniprot_id']}")

    lig_str = (
        ", ".join(f"{l['comp_id']} ({l['name']})" for l in info["ligands"])
        if info["ligands"]
        else "None identified"
    )
    site_str = (
        ", ".join(f"{s['id']}: {s['description']}" for s in info["binding_sites"])
        if info["binding_sites"]
        else "None available"
    )

    text = (
        f"{'=' * 55}\n"
        f"  PROTEIN: {info['title']}\n"
        f"{'=' * 55}\n"
        f"PDB ID         : {info['pdb_id']}\n"
        f"Classification : {info['classification']}\n"
        f"Organism       : {info['organism']}\n"
        f"Resolution     : {info['resolution']}\n"
        f"UniProt ID     : {info.get('uniprot_id') or 'N/A'}\n\n"
        f"LIGANDS: {lig_str}\n\n"
        f"BINDING SITES: {site_str}\n"
        f"{'=' * 55}"
    )

    # AI analysis
    client = _get_openai_client()
    if client:
        try:
            logger.info(f"Requesting AI analysis for {pdb_id}")
            prompt = (
                "You are an expert structural biologist. Briefly analyze this "
                "protein for drug discovery.\n"
                f"PDB: {info['pdb_id']}, Title: {info['title']}, "
                f"Classification: {info['classification']}, "
                f"Organism: {info['organism']}, Ligands: {lig_str}, "
                f"Binding Sites: {site_str}\n\n"
                "Provide a concise analysis: druggable binding sites, key "
                "interactions, and design considerations."
            )
            resp = client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[{"role": "user", "content": prompt}],
            )
            text += "\n\nAI ANALYSIS:\n" + resp.choices[0].message.content
            logger.info(f"AI analysis completed for {pdb_id}")
        except Exception as exc:
            logger.error(f"AI analysis failed for {pdb_id}: {exc}")
            text += f"\n\n[AI analysis failed: {exc}]"

    viewer = _build_3d_viewer_html(pdb_id)
    logger.info(f"=== STEP 1 COMPLETE: {pdb_id} analyzed successfully ===")

    status = f"**Protein:** ✅ Analysis complete for {pdb_id}"
    return status, text, info, viewer


# ── Step 2 — Compound Discovery ────────────────────────────────────────


def _update_compound_status():
    """Show status when compound fetching starts."""
    return "**Compounds:** 🔄 Fetching from ChEMBL (30-60s)..."


def _fetch_compounds(pdb_id):
    """Fetch bioactive compounds from ChEMBL for the protein target."""
    logger.info("=== STEP 2: Fetching compounds ===")

    pdb_id = (pdb_id or "").strip().upper()
    if not pdb_id:
        logger.warning("No PDB ID provided")
        return "**Compounds:** ❌ Enter a PDB ID first", None, None

    # Get UniProt mapping for this protein
    uniprot_id = _get_uniprot_from_pdb(pdb_id)
    logger.info(f"Fetching compounds for {pdb_id} (UniProt: {uniprot_id})")

    compounds: list[dict] = []
    seen_smiles: set[str] = set()
    target_chembl_id = None

    # --- ChEMBL target lookup ---
    if uniprot_id:
        logger.info(f"Looking up ChEMBL target for UniProt {uniprot_id}")
        try:
            r = requests.get(
                "https://www.ebi.ac.uk/chembl/api/data/target.json",
                params={
                    "target_components__accession": uniprot_id,
                    "limit": 1,
                    "format": "json",
                },
                timeout=60,  # ChEMBL API can be slow
            )
            if r.ok:
                targets = r.json().get("targets", [])
                if targets:
                    target_chembl_id = targets[0]["target_chembl_id"]
                    logger.info(f"Found ChEMBL target: {target_chembl_id}")
                else:
                    logger.warning(f"No ChEMBL target found for UniProt {uniprot_id}")
        except Exception as e:
            logger.error(f"Error looking up ChEMBL target: {e}")

    # --- Fetch activities ---
    if target_chembl_id:
        logger.info(f"Fetching activities for ChEMBL target {target_chembl_id}")
        offset = 0
        while offset < 2000:
            logger.debug(f"Fetching activities batch at offset {offset}")
            try:
                r = requests.get(
                    "https://www.ebi.ac.uk/chembl/api/data/activity.json",
                    params={
                        "target_chembl_id": target_chembl_id,
                        "assay_type": "B",
                        "limit": 500,
                        "offset": offset,
                        "format": "json",
                    },
                    timeout=60,  # ChEMBL API can be slow
                )
                if not r.ok:
                    break
                activities = r.json().get("activities", [])
                if not activities:
                    break

                for act in activities:
                    smiles = act.get("canonical_smiles")
                    if not smiles or smiles in seen_smiles:
                        continue
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue

                    val = None
                    try:
                        val = float(act.get("standard_value", 0))
                    except (TypeError, ValueError):
                        pass

                    compounds.append({
                        "smiles": smiles,
                        "name": (
                            act.get("molecule_pref_name")
                            or act.get("molecule_chembl_id", "Unknown")
                        ),
                        "chembl_id": act.get("molecule_chembl_id", ""),
                        "activity_type": act.get("standard_type", ""),
                        "activity_value": val,
                        "activity_units": act.get("standard_units", ""),
                    })
                    seen_smiles.add(smiles)

                offset += 500
                if len(activities) < 500:
                    logger.debug(f"Reached end of activities (got {len(activities)} in last batch)")
                    break
            except Exception as e:
                logger.error(f"Error fetching activities at offset {offset}: {e}")
                break

        logger.info(f"Fetched {len(compounds)} compounds from ChEMBL activities")
    else:
        logger.warning("No ChEMBL target ID available, skipping activity fetch")

    # --- Also grab co-crystallized ligands from the PDB ---
    logger.info(f"Fetching co-crystallized ligands from PDB {pdb_id}")

    # Fetch ligands from PDB structure
    pdb_ligands = []
    try:
        base = "https://data.rcsb.org/rest/v1/core"
        r = requests.get(f"{base}/entry/{pdb_id}", timeout=15)
        if r.ok:
            entry = r.json()
            non_poly_ids = (
                entry.get("rcsb_entry_container_identifiers", {})
                .get("non_polymer_entity_ids") or []
            )
            for eid in non_poly_ids[:10]:
                try:
                    er = requests.get(
                        f"{base}/nonpolymer_entity/{pdb_id}/{eid}", timeout=10
                    )
                    if er.ok:
                        d = er.json().get("pdbx_entity_nonpoly", {})
                        comp_id = d.get("comp_id", "")
                        name = d.get("name", comp_id)
                        if comp_id and comp_id not in _SKIP_LIGANDS:
                            pdb_ligands.append({"comp_id": comp_id, "name": name})
                except Exception as e:
                    logger.debug(f"Error fetching ligand entity {eid}: {e}")
    except Exception as e:
        logger.warning(f"Error fetching PDB ligands: {e}")

    for lig in pdb_ligands:
        try:
            r = requests.get(
                f"https://data.rcsb.org/rest/v1/core/chemcomp/{lig['comp_id']}",
                timeout=10,
            )
            if r.ok:
                desc = r.json().get("rcsb_chem_comp_descriptor", {})
                smi = desc.get("smiles_stereo") or desc.get("smiles")
                if smi and smi not in seen_smiles:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        compounds.append({
                            "smiles": smi,
                            "name": lig.get("name", lig["comp_id"]),
                            "chembl_id": lig["comp_id"],
                            "activity_type": "co-crystallized",
                            "activity_value": None,
                            "activity_units": "",
                        })
                        seen_smiles.add(smi)
        except Exception as e:
            logger.debug(f"Error fetching SMILES for ligand {lig['comp_id']}: {e}")
            continue

    logger.info(f"Added {len(seen_smiles) - (len(compounds) - len(seen_smiles))} PDB ligands")

    if not compounds:
        logger.warning(f"No compounds found for {pdb_id}")
        return (
            f"**Compounds:** ⚠️ None found for {pdb_id}",
            None,
            None,
        )

    # Build display table
    rows = []
    for c in compounds:
        rows.append({
            "Name": c["name"][:30],
            "ChEMBL ID": c["chembl_id"],
            "SMILES": c["smiles"][:50] + ("..." if len(c["smiles"]) > 50 else ""),
            "Assay": c["activity_type"],
            "Value": f"{c['activity_value']:.1f}" if c["activity_value"] else "N/A",
            "Units": c["activity_units"],
        })

    status = f"**Compounds:** ✅ Found {len(compounds)} for {pdb_id}"
    if target_chembl_id:
        status += f" ({target_chembl_id})"

    logger.info(f"=== STEP 2 COMPLETE: {len(compounds)} compounds discovered ===")
    return status, compounds, pd.DataFrame(rows)


# ── Step 3 — Rule of 5 Filter ─────────────────────────────────────────


def _apply_rule_of_5(compounds_state):
    """Filter by Lipinski Rule of 5 with adaptive thresholds."""
    logger.info("=== STEP 3: Applying Rule of 5 filter ===")
    if not compounds_state:
        logger.warning("No compounds provided for Rule of 5 filter")
        return "No compounds to filter. Run Step 2 first.", None, None

    total = len(compounds_state)
    logger.info(f"Filtering {total} compounds by Rule of 5")

    # Calculate properties for all compounds
    for c in compounds_state:
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol is None:
            c["ro5_violations"] = 999  # Mark as invalid
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ])

        c["mw"] = round(mw, 1)
        c["logp"] = round(logp, 2)
        c["hbd"] = hbd
        c["hba"] = hba
        c["ro5_violations"] = violations

    # Remove invalid compounds
    valid_compounds = [c for c in compounds_state if c["ro5_violations"] != 999]

    # Adaptive filtering strategy
    target_min = 100  # Minimum compounds to keep
    target_ratio = 0.5  # Aim for 50% of input

    # Try progressively relaxed thresholds
    for max_violations in [0, 1, 2, 3, 4]:
        candidates = [c for c in valid_compounds if c["ro5_violations"] <= max_violations]

        # If we have enough compounds, use this threshold
        if len(candidates) >= target_min:
            passed = candidates
            logger.info(f"Rule of 5: Using ≤{max_violations} violations threshold")
            break
    else:
        # If even 4 violations doesn't give us enough, just sort by violations
        passed = sorted(valid_compounds, key=lambda x: x["ro5_violations"])[:max(target_min, len(valid_compounds))]
        logger.warning(f"Rule of 5: Relaxed to top {len(passed)} compounds by violations")

    # If we have way more than target, optionally tighten
    target_count = int(total * target_ratio)
    if len(passed) > max(target_count, target_min) and len(passed) > 150:
        # Sort by violations first, then by activity value if available
        passed_sorted = sorted(
            passed,
            key=lambda x: (
                x["ro5_violations"],
                x.get("activity_value") if x.get("activity_value") is not None else float("inf")
            )
        )
        # Keep between target and 150% of target
        keep_count = min(len(passed), max(target_count, target_min))
        passed = passed_sorted[:keep_count]
        logger.info(f"Rule of 5: Tightened to top {len(passed)} compounds")

    # Mark pass/fail
    for c in passed:
        c["ro5_pass"] = True

    rows = []
    for c in passed:
        act = (
            f"{c['activity_value']:.0f} {c['activity_units']}"
            if c.get("activity_value")
            else "N/A"
        )
        rows.append({
            "Name": c["name"][:25],
            "MW": c["mw"],
            "LogP": c["logp"],
            "HBD": c["hbd"],
            "HBA": c["hba"],
            "Violations": c["ro5_violations"],
            "Activity": act,
        })

    # Calculate status message
    pct = (len(passed) / total * 100) if total > 0 else 0
    max_viol = max([c["ro5_violations"] for c in passed]) if passed else 0
    status = f"Rule of 5: {len(passed)} of {total} compounds ({pct:.0f}%) — max {max_viol} violations"

    df = pd.DataFrame(rows) if rows else None
    logger.info(f"=== STEP 3 COMPLETE: {len(passed)}/{total} compounds passed Rule of 5 ===")
    return status, passed, df


# ── Step 4 — Binding Activity Ranking ──────────────────────────────────


def _calculate_optimal_workers() -> int:
    """Calculate optimal number of parallel workers based on available resources."""
    import multiprocessing

    # Get CPU cores (use 'spawn' context to avoid fork issues with Gradio)
    total_cpus = multiprocessing.cpu_count()

    # Try to get available RAM
    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().available / (1024**3)

        # Estimate: Each Vina process uses ~300-500MB RAM
        # Use conservative 500MB estimate
        ram_based_workers = int(available_ram_gb / 0.5)

        # Also check current CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        # If CPU is already busy, reduce workers
        if cpu_usage > 50:
            available_cpu_fraction = (100 - cpu_usage) / 100
            cpu_based_workers = max(1, int(total_cpus * available_cpu_fraction))
        else:
            # Leave 1-2 cores for system
            cpu_based_workers = max(1, total_cpus - 2)

        # Use minimum of RAM and CPU constraints
        optimal = min(ram_based_workers, cpu_based_workers)

        logger.info(
            f"Resource detection: {total_cpus} CPUs, {available_ram_gb:.1f}GB RAM available, "
            f"CPU usage {cpu_usage:.0f}% → {optimal} workers"
        )

    except ImportError:
        # psutil not available, use conservative estimate
        # Leave 2 cores for system, use conservative estimate
        optimal = max(1, total_cpus - 2)
        logger.info(
            f"psutil not available, using conservative estimate: "
            f"{total_cpus} CPUs → {optimal} workers"
        )

    # Ensure at least 1 worker, cap at 32 for stability
    return max(1, min(optimal, 32))


def _fetch_pdb_file(pdb_id: str) -> str:
    """Fetch PDB file text from RCSB."""
    pdb_id = pdb_id.strip().upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except HTTPError as e:
        if e.code == 404:
            raise Exception(f"PDB ID '{pdb_id}' not found on RCSB.")
        raise Exception(f"Failed to fetch PDB '{pdb_id}': HTTP {e.code}")


def _find_binding_center(pdb_text: str) -> tuple[float, float, float]:
    """Find binding site center from co-crystallized ligand."""
    skip = {"HOH", "WAT", "SO4", "PO4", "GOL", "EDO", "PEG", "DMS", "ACT",
            "CL", "NA", "MG", "ZN", "CA", "K", "FMT", "IOD"}
    residue_coords = {}

    for line in pdb_text.splitlines():
        if not line.startswith("HETATM"):
            continue
        res_name = line[17:20].strip()
        if res_name in skip:
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            residue_coords.setdefault(res_name, []).append((x, y, z))
        except (ValueError, IndexError):
            continue

    if not residue_coords:
        # Fallback: use geometric center of protein
        logger.warning("No ligand found, using protein geometric center")
        return (0.0, 0.0, 0.0)

    best_res = max(residue_coords, key=lambda r: len(residue_coords[r]))
    coords = np.array(residue_coords[best_res])
    center = coords.mean(axis=0)
    return round(float(center[0]), 3), round(float(center[1]), 3), round(float(center[2]), 3)


def _prepare_receptor_pdbqt(pdb_path: str, output_path: str) -> None:
    """Prepare receptor PDBQT from PDB file."""
    # Clean PDB (remove HETATM)
    cleaned_path = pdb_path.replace(".pdb", "_clean.pdb")
    with open(pdb_path) as f:
        lines = f.readlines()
    with open(cleaned_path, "w") as f:
        for line in lines:
            if line.startswith(("ATOM", "TER", "END", "MODEL", "ENDMDL", "REMARK", "CRYST1")):
                f.write(line)

    # Run mk_prepare_receptor
    mk_script = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")
    if mk_script is None:
        raise Exception("mk_prepare_receptor not found. Install with: pip install meeko")

    output_basename = output_path.removesuffix(".pdbqt")
    cmd = [mk_script, "-i", cleaned_path, "-o", output_basename, "-p"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise Exception(f"Receptor prep failed: {result.stderr[:200]}")


# Note: _smiles_to_pdbqt and dock_single_compound moved to docking_worker.py module


def _rank_by_activity(ro5_state, protein_info_state=None, num_dock=50):
    """Rank compounds using AutoDock Vina docking scores."""
    logger.info("=== STEP 4: Ranking by binding activity (AutoDock Vina) ===")
    if not ro5_state:
        logger.warning("No compounds provided for docking")
        return "No compounds to rank. Run Step 3 first.", None, None

    if not protein_info_state:
        logger.error("No protein info for docking")
        return "Protein info required for docking. Re-run Step 1.", None, None

    total = len(ro5_state)
    pdb_id = protein_info_state["pdb_id"]

    # Pre-rank by ChEMBL activity to select top candidates
    with_activity = [c for c in ro5_state if c.get("activity_value") and c["activity_value"] > 0]
    without_activity = [c for c in ro5_state if not c.get("activity_value") or c["activity_value"] <= 0]

    with_activity.sort(key=lambda x: x["activity_value"])
    pre_ranked = with_activity + without_activity

    # Dock top N compounds
    num_dock = min(num_dock, len(pre_ranked))
    to_dock = pre_ranked[:num_dock]

    # Calculate optimal workers based on available resources
    num_workers = min(_calculate_optimal_workers(), num_dock)
    est_time = (num_dock / num_workers) * 0.5  # ~30s per compound per worker
    logger.info(f"Docking {num_dock} compounds in parallel ({num_workers} workers) - ETA: {est_time:.1f}min")

    try:
        # Fetch PDB and prepare receptor (once)
        pdb_text = _fetch_pdb_file(pdb_id)
        center = _find_binding_center(pdb_text)
        logger.info(f"Binding center: {center}")

        # Prepare receptor in a persistent temp directory
        tmpdir = tempfile.mkdtemp()
        try:
            pdb_path = os.path.join(tmpdir, "receptor.pdb")
            with open(pdb_path, "w") as f:
                f.write(pdb_text)

            receptor_pdbqt = os.path.join(tmpdir, "receptor.pdbqt")
            _prepare_receptor_pdbqt(pdb_path, receptor_pdbqt)
            logger.info("Receptor prepared, starting parallel docking...")

            # Prepare arguments for parallel docking
            dock_args = [
                (i, to_dock[i]["smiles"], receptor_pdbqt, center)
                for i in range(num_dock)
            ]

            # Run docking in parallel using isolated worker module
            completed = 0
            failed = 0
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(dock_single_compound, arg): arg[0] for arg in dock_args}

                for future in as_completed(futures):
                    idx, affinity = future.result()
                    to_dock[idx]["vina_affinity"] = round(affinity, 2)
                    to_dock[idx]["docked"] = True

                    if affinity == float('inf'):
                        failed += 1

                    completed += 1
                    # Progress every 5 compounds or at completion
                    if completed % 5 == 0 or completed == num_dock:
                        pct = (completed / num_dock) * 100
                        logger.info(f"Docking progress: {completed}/{num_dock} ({pct:.0f}%) - {failed} failed")
                        print(f"[DOCKING PROGRESS] {completed}/{num_dock} ({pct:.0f}%) - {failed} failed", flush=True)

            logger.info(f"Parallel docking complete: {num_dock} compounds ({failed} failed)")

        finally:
            # Clean up temp directory
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Docking failed: {e}")
        return f"Docking failed: {e}", None, None

    # Keep ONLY successfully docked compounds (discard undocked and failed)
    docked = [
        c for c in pre_ranked
        if c.get("docked") and c.get("vina_affinity") != float('inf')
    ]

    # Sort by Vina affinity (lower = better binding)
    docked.sort(key=lambda x: x["vina_affinity"])
    ranked = docked  # Only docked compounds - no undocked!

    for i, c in enumerate(ranked):
        c["binding_rank"] = i + 1

    rows = []
    for c in ranked[:500]:
        rows.append({
            "Rank": c["binding_rank"],
            "Name": c["name"][:25],
            "Vina (kcal/mol)": (
                f"{c['vina_affinity']:.2f}"
                if c.get("vina_affinity") and c["vina_affinity"] != float('inf')
                else "Not docked"
            ),
            "ChEMBL Activity": (
                f"{c['activity_value']:.0f} {c.get('activity_units', '')}"
                if c.get("activity_value")
                else "N/A"
            ),
            "MW": c.get("mw", ""),
            "LogP": c.get("logp", ""),
        })

    successful_docks = len(ranked)
    failed_docks = num_dock - successful_docks
    status = f"Successfully docked {successful_docks}/{num_dock} compounds with AutoDock Vina"
    if failed_docks > 0:
        status += f" ({failed_docks} failed and discarded)"

    df = pd.DataFrame(rows) if rows else None
    logger.info(f"=== STEP 4 COMPLETE: {successful_docks} successful, {failed_docks} failed ===")
    return status, ranked, df


# ── Step 5 — ADME Filter ──────────────────────────────────────────────


def _apply_adme_filter(docked_state):
    """Filter by ADME criteria using RDKit descriptors with adaptive thresholds."""
    logger.info("=== STEP 5: Applying ADME filter ===")
    if not docked_state:
        logger.warning("No compounds provided for ADME filter")
        return "No compounds to filter. Run Step 4 first.", None, None

    total = len(docked_state)
    logger.info(f"Filtering {total} compounds by ADME criteria")

    # Calculate ADME properties for all compounds
    valid_compounds = []
    for c in docked_state:
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol is None:
            continue

        tpsa = Descriptors.TPSA(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        mw = c.get("mw") or Descriptors.MolWt(mol)
        logp = c.get("logp") or Descriptors.MolLogP(mol)

        criteria = {
            "TPSA < 140": tpsa < 140,
            "RotBonds <= 10": rot_bonds <= 10,
            "MW 150-500": 150 <= mw <= 500,
            "LogP -0.4 to 5.6": -0.4 <= logp <= 5.6,
            "HBD <= 5": c.get("hbd", 0) <= 5,
            "HBA <= 10": c.get("hba", 0) <= 10,
        }
        score = sum(criteria.values())

        c["tpsa"] = round(tpsa, 1)
        c["rot_bonds"] = rot_bonds
        c["adme_score"] = score

        valid_compounds.append(c)

    # Adaptive filtering strategy
    target_min = 50  # Minimum compounds to keep
    target_ratio = 0.5  # Aim for 50% of input

    # Try progressively relaxed thresholds (6/6 down to 0/6)
    for min_score in [6, 5, 4, 3, 2, 1, 0]:
        candidates = [c for c in valid_compounds if c["adme_score"] >= min_score]

        # If we have enough compounds, use this threshold
        if len(candidates) >= target_min:
            passed = candidates
            logger.info(f"ADME: Using >={min_score}/6 criteria threshold")
            break
    else:
        # Fallback: take all valid compounds
        passed = valid_compounds
        logger.warning(f"ADME: Using all {len(passed)} compounds")

    # If we have way more than target, tighten by taking top by ADME score + binding
    target_count = max(int(total * target_ratio), target_min)
    if len(passed) > target_count:
        # Sort by ADME score (higher better), then by Vina affinity (lower better)
        passed_sorted = sorted(
            passed,
            key=lambda x: (
                -x["adme_score"],  # Higher ADME score is better
                x.get("vina_affinity", float('inf'))  # Lower affinity is better
            )
        )
        passed = passed_sorted[:target_count]
        logger.info(f"ADME: Tightened to top {len(passed)} compounds")

    # Mark pass/fail
    for c in passed:
        c["adme_pass"] = True

    # Build output table
    rows = []
    for c in passed:
        rows.append({
            "Rank": c.get("binding_rank", ""),
            "Name": c["name"][:25],
            "ADME": f"{c['adme_score']}/6",
            "Vina": (
                f"{c.get('vina_affinity', 'N/A'):.2f}"
                if c.get("vina_affinity") and c["vina_affinity"] != float('inf')
                else "N/A"
            ),
            "TPSA": c["tpsa"],
            "RotBonds": c["rot_bonds"],
            "MW": c.get("mw", ""),
            "LogP": c.get("logp", ""),
        })

    # Calculate status message
    pct = (len(passed) / total * 100) if total > 0 else 0
    min_score = min([c["adme_score"] for c in passed]) if passed else 0
    max_score = max([c["adme_score"] for c in passed]) if passed else 0
    status = (
        f"ADME: {len(passed)} of {total} compounds ({pct:.0f}%) — "
        f"scores {min_score}-{max_score}/6"
    )

    df = pd.DataFrame(rows) if rows else None
    logger.info(f"=== STEP 5 COMPLETE: {len(passed)}/{total} compounds passed ADME filter ===")
    return status, passed, df


# ── Step 6 — Compound Detail ──────────────────────────────────────────


def _on_select_compound(evt: gr.SelectData, final_state):
    """Handle click on a row in the final results table."""
    if not final_state:
        logger.warning("Compound selection attempted with no compounds available")
        return "No compound selected.", None, None
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx >= len(final_state):
        logger.warning(f"Invalid compound index selected: {idx}")
        return "Invalid selection.", None, None

    compound = final_state[idx]
    smiles = compound["smiles"]
    logger.info(f"Selected compound: {compound.get('name')} (ChEMBL: {compound.get('chembl_id')})")

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

    return detail, img, compound


def _ai_explain_compound(selected_state, protein_state):
    """AI-generated explanation of how the compound interacts with the protein."""
    logger.info("=== Generating AI compound explanation ===")
    if not selected_state:
        logger.warning("AI explanation requested with no compound selected")
        return "Select a compound from the results table first."
    if not protein_state:
        logger.warning("AI explanation requested with no protein data")
        return "No protein data available."

    client = _get_openai_client()
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


def _populate_compound_selector(final_state):
    """Populate dropdown with compound choices after ADME filter."""
    if not final_state:
        return gr.Dropdown(choices=[], value=None)

    # Create choices as list of tuples: (display_name, index)
    choices = []
    for i, compound in enumerate(final_state):
        # Display format: "Rank 1: CHEMBL123 (Name) - Vina: -8.5"
        vina_score = compound.get("vina_affinity", "N/A")
        vina_str = f"{vina_score:.2f}" if isinstance(vina_score, (int, float)) and vina_score != float('inf') else "N/A"

        display_name = (
            f"Rank {compound.get('binding_rank', i+1)}: "
            f"{compound.get('chembl_id', 'N/A')} ({compound.get('name', 'Unknown')[:30]}) - "
            f"Vina: {vina_str} kcal/mol, ADME: {compound.get('adme_score', 'N/A')}/6"
        )
        choices.append((display_name, i))

    logger.info(f"Populated compound selector with {len(choices)} compounds")
    return gr.Dropdown(choices=choices, value=None if not choices else choices[0][1])


def _on_compound_selected(selected_idx, final_state):
    """Handle compound selection from dropdown."""
    if final_state is None or selected_idx is None:
        logger.warning("Compound selection attempted with no compounds or index")
        return "No compound selected.", None, None

    if selected_idx >= len(final_state):
        logger.warning(f"Invalid compound index selected: {selected_idx}")
        return "Invalid selection.", None, None

    compound = final_state[selected_idx]
    smiles = compound["smiles"]
    logger.info(f"Selected compound: {compound.get('name')} (ChEMBL: {compound.get('chembl_id')})")

    # Build detailed information text
    detail = (
        f"{'=' * 50}\n"
        f"  COMPOUND: {compound['name']}\n"
        f"{'=' * 50}\n"
        f"ChEMBL ID        : {compound.get('chembl_id', 'N/A')}\n"
        f"SMILES           : {smiles}\n\n"
        f"PROPERTIES:\n"
        f"  Molecular Weight : {compound.get('mw', 'N/A')}\n"
        f"  LogP             : {compound.get('logp', 'N/A')}\n"
        f"  H-Bond Donors    : {compound.get('hbd', 'N/A')}\n"
        f"  H-Bond Acceptors : {compound.get('hba', 'N/A')}\n"
        f"  TPSA             : {compound.get('tpsa', 'N/A')}\n"
        f"  Rotatable Bonds  : {compound.get('rot_bonds', 'N/A')}\n\n"
        f"BINDING DATA:\n"
        f"  Vina Affinity : {compound.get('vina_affinity', 'N/A')} kcal/mol\n"
        f"  Assay Type    : {compound.get('activity_type', 'N/A')}\n"
        f"  Value         : {compound.get('activity_value', 'N/A')} "
        f"{compound.get('activity_units', '')}\n\n"
        f"FILTERS:\n"
        f"  Rule of 5  : "
        f"{'Pass' if compound.get('ro5_pass') else 'Fail'} "
        f"({compound.get('ro5_violations', '?')} violations)\n"
        f"  ADME Score : {compound.get('adme_score', 'N/A')}/6\n"
        f"{'=' * 50}"
    )

    # Generate 2D structure image
    mol = Chem.MolFromSmiles(smiles)
    img = None
    if mol:
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(400, 400))

    return detail, img, compound


def _ai_explain_protein(protein_state):
    """AI-generated deep dive on the protein target."""
    logger.info("=== Generating AI protein deep dive ===")
    if not protein_state:
        logger.warning("Protein deep dive requested with no protein data")
        return "No protein data available. Run Step 1 first."

    client = _get_openai_client()
    if not client:
        logger.warning("Protein deep dive requested but OpenAI API key not set")
        return "[OpenAI API key not set]"

    p = protein_state
    logger.info(f"Generating protein deep dive for {p['pdb_id']}")
    lig_str = (
        ", ".join(f"{l['comp_id']} ({l['name']})" for l in p.get("ligands", []))
        or "none"
    )

    prompt = (
        "You are an expert structural biologist and medicinal chemist.\n"
        f"Provide a detailed analysis of this protein target:\n\n"
        f"PDB ID: {p['pdb_id']}\n"
        f"Title: {p['title']}\n"
        f"Classification: {p['classification']}\n"
        f"Organism: {p['organism']}\n"
        f"Resolution: {p['resolution']}\n"
        f"Ligands: {lig_str}\n\n"
        "Cover:\n"
        "1. BINDING SITES — location and functional significance\n"
        "2. KEY RESIDUES — catalytic, allosteric, drug-interacting\n"
        "3. INTERACTION TYPES — H-bonds, hydrophobic, electrostatic, etc.\n"
        "4. DESIGN CONSIDERATIONS for drug/inhibitor design\n"
        "5. DRUG DEVELOPMENT INSIGHTS — druggability, selectivity, "
        "known inhibitor classes\n\n"
        "Be specific and concise."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": prompt}],
        )
        logger.info("AI protein deep dive completed successfully")
        return resp.choices[0].message.content
    except Exception as exc:
        logger.error(f"AI protein deep dive failed: {exc}")
        return f"AI analysis failed: {exc}"


# ── UI ─────────────────────────────────────────────────────────────────


def create_tab():
    with gr.Tab("Drug Discovery Pipeline"):
        gr.Markdown(
            "# Integrated Drug Discovery Pipeline\n"
        )

        # --- State ---
        protein_state = gr.State(None)
        all_compounds_state = gr.State(None)
        ro5_state = gr.State(None)
        ranked_state = gr.State(None)
        final_state = gr.State(None)
        selected_state = gr.State(None)

        # ────────────────────────────────────────────────────────
        # Main Input & Run Button
        # ────────────────────────────────────────────────────────
        gr.Markdown(
            "## Quick Start: Run Complete Pipeline\n"
            "Enter a PDB ID and click the button below to run the entire pipeline automatically."
        )

        with gr.Row():
            pdb_input = gr.Textbox(
                label="PDB ID",
                placeholder="e.g., 6LU7",
                lines=1,
                scale=2,
            )
            num_dock_input = gr.Number(
                label="Compounds to Dock",
                value=20,
                minimum=1,
                maximum=500,
                step=1,
                scale=1,
            )

        run_pipeline_btn = gr.Button(
            "🚀 Run Complete Pipeline",
            variant="primary",
            size="lg",
        )

        gr.Examples(
            examples=[["6LU7"], ["1IEP"], ["2HYY"], ["1AZ5"]],
            inputs=pdb_input,
        )

        gr.Markdown(
            "_Pipeline: Protein Analysis + Compound Discovery → Rule of 5 → "
            "AutoDock Vina Docking → ADME Filter → Results_"
        )

        # ────────────────────────────────────────────────────────
        # Pipeline Progress & Intermediate Results
        # ────────────────────────────────────────────────────────
        gr.Markdown("---\n## Pipeline Progress")

        # Step 1 & 2 Results
        with gr.Accordion("Step 1 & 2: Protein Details & Discovered Compounds", open=True):
            with gr.Row():
                protein_status = gr.Markdown(
                    value="**Protein:** Not started",
                    visible=True,
                )
                compounds_status = gr.Markdown(
                    value="**Compounds:** Not started",
                    visible=True,
                )

            with gr.Accordion("Protein Information & 3D Structure", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        protein_text = gr.Textbox(
                            label="Protein Information & AI Analysis",
                            interactive=False,
                            lines=20,
                        )
                    with gr.Column(scale=1):
                        viewer_html = gr.HTML(value=_VIEWER_PLACEHOLDER)

            with gr.Accordion("All Discovered Compounds (Unfiltered)", open=False):
                compounds_table = gr.Dataframe(
                    label="Compounds from ChEMBL",
                    interactive=False,
                    wrap=True,
                    max_height=300,
                )

        # Pipeline filter status
        ro5_status = gr.Textbox(
            label="Step 3: Rule of 5", interactive=False, lines=1, value="Not started"
        )
        rank_status = gr.Textbox(
            label="Step 4: AutoDock Vina Docking", interactive=False, lines=1, value="Not started"
        )
        adme_status = gr.Textbox(
            label="Step 5: ADME Filter", interactive=False, lines=1, value="Not started"
        )

        # ────────────────────────────────────────────────────────
        # Results — Compound Detail
        # ────────────────────────────────────────────────────────
        gr.Markdown(
            "---\n## Final Results: Select Compound to View Details"
        )

        compound_selector = gr.Dropdown(
            label="Select Compound to View Details",
            choices=[],
            value=None,
            interactive=True,
            scale=2,
        )

        with gr.Row():
            with gr.Column(scale=1):
                detail_text = gr.Textbox(
                    label="Compound Properties",
                    interactive=False,
                    lines=22,
                )
            with gr.Column(scale=1):
                detail_image = gr.Image(
                    label="2D Structure",
                    type="pil",
                    height=400,
                )
        explain_compound_btn = gr.Button(
            "AI: Explain This Compound", variant="secondary", size="lg"
        )
        compound_explanation = gr.Textbox(
            label="AI Compound Analysis",
            interactive=False,
            lines=15,
        )

        # ────────────────────────────────────────────────────────
        # Advanced Manual Control (Optional)
        # ────────────────────────────────────────────────────────
        with gr.Accordion("⚙️ Advanced: Manual Step Control & Detailed Tables", open=False):
            gr.Markdown(
                "Use these buttons to run individual pipeline steps manually. "
                "Results appear in the Pipeline Progress section above."
            )

            gr.Markdown("### Step 1 & 2: Protein Analysis + Compound Discovery")
            analyze_btn = gr.Button(
                "Run Step 1 & 2 Only",
                variant="secondary",
                size="sm",
            )

            gr.Markdown("### Step 3: Rule of 5 Filter")
            ro5_btn = gr.Button("Run Rule of 5 Filter", variant="secondary", size="sm")
            with gr.Accordion("Intermediate: Rule of 5 Results Table", open=False):
                ro5_table = gr.Dataframe(interactive=False, max_height=250)

            gr.Markdown("### Step 4: AutoDock Vina Docking")
            rank_btn = gr.Button("Run Vina Docking", variant="secondary", size="sm")
            with gr.Accordion("Intermediate: Docking Results Table", open=False):
                rank_table = gr.Dataframe(interactive=False, max_height=250)

            gr.Markdown("### Step 5: ADME Filter")
            adme_btn = gr.Button("Run ADME Filter", variant="secondary", size="sm")
            with gr.Accordion("Intermediate: ADME Results Table", open=False):
                adme_table = gr.Dataframe(interactive=False, max_height=250)

        # ────────────────────────────────────────────────────────
        # Wire events
        # ────────────────────────────────────────────────────────

        # Main "Run Complete Pipeline" button
        run_pipeline_btn.click(
            _run_complete_pipeline,
            inputs=[pdb_input, num_dock_input],
            outputs=[
                protein_status,
                protein_text,
                protein_state,
                viewer_html,
                compounds_status,
                all_compounds_state,
                compounds_table,
                ro5_status,
                ro5_state,
                ro5_table,
                rank_status,
                ranked_state,
                rank_table,
                adme_status,
                final_state,
                adme_table,
                compound_selector,
            ],
        )

        # Advanced: Individual step buttons
        analyze_btn.click(
            _run_protein_and_compounds_parallel,
            inputs=[pdb_input],
            outputs=[
                protein_status,
                protein_text,
                protein_state,
                viewer_html,
                compounds_status,
                all_compounds_state,
                compounds_table,
            ],
        )

        # Advanced: Step 3
        ro5_btn.click(
            _apply_rule_of_5,
            inputs=[all_compounds_state],
            outputs=[ro5_status, ro5_state, ro5_table],
        )

        # Step 4
        rank_btn.click(
            _rank_by_activity,
            inputs=[ro5_state, protein_state, num_dock_input],
            outputs=[rank_status, ranked_state, rank_table],
        )

        # Step 5
        adme_btn.click(
            _apply_adme_filter,
            inputs=[ranked_state],
            outputs=[adme_status, final_state, adme_table],
        ).then(
            _populate_compound_selector,
            inputs=[final_state],
            outputs=[compound_selector],
        )

        # Compound selection & detail (dropdown selector)
        compound_selector.change(
            _on_compound_selected,
            inputs=[compound_selector, final_state],
            outputs=[detail_text, detail_image, selected_state],
        )

        # Alternative: table row selection (still works)
        adme_table.select(
            _on_select_compound,
            inputs=[final_state],
            outputs=[detail_text, detail_image, selected_state],
        )

        # AI explanation
        explain_compound_btn.click(
            _ai_explain_compound,
            inputs=[selected_state, protein_state],
            outputs=[compound_explanation],
        )
