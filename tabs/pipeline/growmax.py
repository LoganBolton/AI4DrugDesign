"""GrowMax — fragment-based molecule growing for drug candidate generation."""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from tabs.pipeline.helpers import logger

_REACTION_SMARTS = [
    ("methyl",       "[c,C:1][H]>>[c,C:1]C"),
    ("methoxy",      "[c,C:1][H]>>[c,C:1]OC"),
    ("amino",        "[c,C:1][H]>>[c,C:1]N"),
    ("hydroxyl",     "[c,C:1][H]>>[c,C:1]O"),
    ("fluoro",       "[c,C:1][H]>>[c,C:1]F"),
    ("chloro",       "[c,C:1][H]>>[c,C:1]Cl"),
    ("cyano",        "[c,C:1][H]>>[c,C:1]C#N"),
    ("amide",        "[c,C:1][H]>>[c,C:1]C(=O)N"),
    ("carboxyl",     "[c,C:1][H]>>[c,C:1]C(=O)O"),
    ("CF3",          "[c,C:1][H]>>[c,C:1]C(F)(F)F"),
    ("phenyl",       "[c,C:1][H]>>[c,C:1]c1ccccc1"),
    ("pyridyl",      "[c,C:1][H]>>[c,C:1]c1ccncc1"),
    ("piperidyl",    "[c,C:1][H]>>[c,C:1]C1CCNCC1"),
    ("morpholinyl",  "[c,C:1][H]>>[c,C:1]C1CCOCC1"),
    ("sulfonamide",  "[c,C:1][H]>>[c,C:1]S(=O)(=O)N"),
    ("acetamide",    "[N:1][H]>>[N:1]C(=O)C"),
]

_COMPILED_REACTIONS = [
    (label, AllChem.ReactionFromSmarts(smarts))
    for label, smarts in _REACTION_SMARTS
]


def _is_drug_like(mol) -> bool:
    """Loose drug-likeness gate to prune clearly non-drug variants before docking."""
    return (
        Descriptors.MolWt(mol) <= 550
        and Descriptors.MolLogP(mol) <= 6
        and Descriptors.NumHDonors(mol) <= 5
        and Descriptors.NumHAcceptors(mol) <= 11
    )


def generate_variants(seed_smiles: str, max_variants: int = 200) -> list[tuple[str, str]]:
    """
    Enumerate grown molecules by replacing one H on the seed with each growth fragment.

    Returns a list of (canonical_smiles, fragment_label) tuples, at most max_variants.
    """
    seed = Chem.MolFromSmiles(seed_smiles)
    if seed is None:
        raise ValueError(f"Invalid SMILES: {seed_smiles}")

    seen = {Chem.MolToSmiles(seed)}
    variants: list[tuple[str, str]] = []

    for label, rxn in _COMPILED_REACTIONS:
        try:
            products = rxn.RunReactants((seed,))
        except Exception:
            continue

        for prod_tuple in products:
            if not prod_tuple:
                continue
            prod = prod_tuple[0]
            try:
                Chem.SanitizeMol(prod)
            except Exception:
                continue
            if not _is_drug_like(prod):
                continue
            canon = Chem.MolToSmiles(prod)
            if canon in seen:
                continue
            seen.add(canon)
            variants.append((canon, label))
            if len(variants) >= max_variants:
                return variants

    return variants


def fetch_growmax_compounds(seed_smiles: str, max_variants: int = 200):
    """
    Generate drug candidates by growing a seed fragment.

    Returns (status, compounds, df) — identical format to fetch_compounds so
    the rest of the pipeline (Ro5 → Vina → ADME) is reused unchanged.
    """
    logger.info("=== STEP 2 (GrowMax): Growing fragment variants ===")

    seed_smiles = (seed_smiles or "").strip()
    if not seed_smiles:
        return "**Step 2 – GrowMax:** ❌ Enter a seed SMILES first", None, None

    if Chem.MolFromSmiles(seed_smiles) is None:
        return "**Step 2 – GrowMax:** ❌ Invalid SMILES string", None, None

    try:
        variants = generate_variants(seed_smiles, max_variants)
    except Exception as exc:
        logger.exception(f"GrowMax variant generation failed: {exc}")
        return f"**Step 2 – GrowMax:** ❌ {exc}", None, None

    if not variants:
        return (
            "**Step 2 – GrowMax:** ⚠️ No valid variants generated — "
            "try a different seed fragment",
            None,
            None,
        )

    compounds = [
        {
            "smiles": smi,
            "name": f"{label}_{i + 1}",
            "chembl_id": "GrowMax",
            "activity_type": "fragment-grown",
            "activity_value": None,
            "activity_units": "",
        }
        for i, (smi, label) in enumerate(variants)
    ]

    rows = [
        {
            "Name": c["name"],
            "Added Fragment": c["name"].rsplit("_", 1)[0],
            "SMILES": c["smiles"][:60] + ("..." if len(c["smiles"]) > 60 else ""),
        }
        for c in compounds
    ]

    status = (
        f"**Step 2 – GrowMax:** ✅ Generated {len(compounds)} variants from seed fragment"
    )
    logger.info(f"=== STEP 2 (GrowMax) COMPLETE: {len(compounds)} variants ===")
    return status, compounds, pd.DataFrame(rows)
