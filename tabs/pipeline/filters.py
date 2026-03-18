"""Steps 3 & 5 — Rule of 5 and ADME filters."""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from tabs.pipeline.helpers import logger


def apply_rule_of_5(compounds_state):
    """Filter by Lipinski Rule of 5 with adaptive thresholds."""
    logger.info("=== STEP 3: Applying Rule of 5 filter ===")
    if not compounds_state:
        logger.warning("No compounds provided for Rule of 5 filter")
        return "**Step 3 – Rule of 5:** ❌ No compounds to filter", None, None

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

        violations = sum((
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ))

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
    status = f"**Step 3 – Rule of 5:** ✅ {len(passed)} of {total} compounds ({pct:.0f}%) — max {max_viol} violations"

    df = pd.DataFrame(rows) if rows else None
    logger.info(f"=== STEP 3 COMPLETE: {len(passed)}/{total} compounds passed Rule of 5 ===")
    return status, passed, df


def apply_adme_filter(docked_state):
    """Filter by ADME criteria using RDKit descriptors with adaptive thresholds."""
    logger.info("=== STEP 5: Applying ADME filter ===")
    if not docked_state:
        logger.warning("No compounds provided for ADME filter")
        return "**Step 5 – ADME:** ❌ No compounds to filter", None, None

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
        f"**Step 5 – ADME:** ✅ {len(passed)} of {total} compounds ({pct:.0f}%) — "
        f"scores {min_score}-{max_score}/6"
    )

    df = pd.DataFrame(rows) if rows else None
    logger.info(f"=== STEP 5 COMPLETE: {len(passed)}/{total} compounds passed ADME filter ===")
    return status, passed, df
