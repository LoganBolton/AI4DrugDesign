"""Step 2 — Compound discovery from ChEMBL and PDB ligands."""

import pandas as pd
import requests
from rdkit import Chem

from tabs.pipeline.helpers import fetch_pdb_ligands, get_uniprot_from_pdb, logger


def fetch_compounds(pdb_id):
    """Fetch bioactive compounds from ChEMBL for the protein target."""
    logger.info("=== STEP 2: Fetching compounds ===")

    pdb_id = (pdb_id or "").strip().upper()
    if not pdb_id:
        logger.warning("No PDB ID provided")
        return "**Step 2 – Compounds:** ❌ Enter a PDB ID first", None, None

    # Get UniProt mapping for this protein
    uniprot_id = get_uniprot_from_pdb(pdb_id)
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
                timeout=60,
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
                    timeout=60,
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
    pdb_ligands = fetch_pdb_ligands(pdb_id)
    chembl_count = len(compounds)

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

    logger.info(f"Added {len(compounds) - chembl_count} PDB ligands")

    if not compounds:
        logger.warning(f"No compounds found for {pdb_id}")
        return (
            f"**Step 2 – Compounds:** ⚠️ None found for {pdb_id}",
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

    status = f"**Step 2 – Compounds:** ✅ Found {len(compounds)} for {pdb_id}"
    if target_chembl_id:
        status += f" ({target_chembl_id})"

    logger.info(f"=== STEP 2 COMPLETE: {len(compounds)} compounds discovered ===")
    return status, compounds, pd.DataFrame(rows)
