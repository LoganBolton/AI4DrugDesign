"""GrowMax — fragment-based molecule growing for drug candidate generation."""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdChemReactions import ReactionFromSmarts

from tabs.pipeline.helpers import logger

# [c;H1:1] matches an aromatic carbon with exactly 1 implicit H — no explicit H needed.
# [N;H1:1] matches N with 1 implicit H (pyrrole-type NH, piperidine NH, etc.).
_REACTION_SMARTS = [
    ("methyl",      "[c;H1:1]>>[c:1]C"),
    ("methoxy",     "[c;H1:1]>>[c:1]OC"),
    ("amino",       "[c;H1:1]>>[c:1]N"),
    ("hydroxyl",    "[c;H1:1]>>[c:1]O"),
    ("fluoro",      "[c;H1:1]>>[c:1]F"),
    ("chloro",      "[c;H1:1]>>[c:1]Cl"),
    ("cyano",       "[c;H1:1]>>[c:1]C#N"),
    ("amide",       "[c;H1:1]>>[c:1]C(=O)N"),
    ("carboxyl",    "[c;H1:1]>>[c:1]C(=O)O"),
    ("CF3",         "[c;H1:1]>>[c:1]C(F)(F)F"),
    ("phenyl",      "[c;H1:1]>>[c:1]c1ccccc1"),
    ("pyridyl",     "[c;H1:1]>>[c:1]c1ccncc1"),
    ("piperidyl",   "[c;H1:1]>>[c:1]C1CCNCC1"),
    ("morpholinyl", "[c;H1:1]>>[c:1]C1CCOCC1"),
    ("sulfonamide", "[c;H1:1]>>[c:1]S(=O)(=O)N"),
    ("acetamide",   "[N;H1:1]>>[N:1]C(=O)C"),
]

_COMPILED_REACTIONS = [
    (label, ReactionFromSmarts(smarts))
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


def _grow_one_round(parent_smiles_list, seen, max_variants, round_num):
    """Grow every molecule in parent_smiles_list by one step. Returns new (smiles, label) pairs."""
    new_variants: list[tuple[str, str]] = []
    label_suffix = f"_r{round_num}"

    for parent_smiles in parent_smiles_list:
        parent = Chem.MolFromSmiles(parent_smiles)
        if parent is None:
            continue
        for label, rxn in _COMPILED_REACTIONS:
            try:
                products = rxn.RunReactants((parent,))
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
                new_variants.append((canon, label + label_suffix))
                if len(new_variants) >= max_variants:
                    return new_variants

    return new_variants


def generate_variants(
    seed_smiles: str, max_variants: int = 200, rounds: int = 2
) -> list[tuple[str, str]]:
    """
    Iteratively grow the seed fragment for `rounds` steps.

    Each round takes the previous round's new variants as parents and grows them
    one step further. Returns at most max_variants (canonical_smiles, label) tuples.
    """
    seed = Chem.MolFromSmiles(seed_smiles)
    if seed is None:
        raise ValueError(f"Invalid SMILES: {seed_smiles}")

    seen: set[str] = {Chem.MolToSmiles(seed)}
    all_variants: list[tuple[str, str]] = []
    current_gen = [Chem.MolToSmiles(seed)]

    for round_num in range(1, rounds + 1):
        remaining = max_variants - len(all_variants)
        if remaining <= 0:
            break
        new = _grow_one_round(current_gen, seen, remaining, round_num)
        all_variants.extend(new)
        current_gen = [smi for smi, _ in new]
        logger.info(f"GrowMax round {round_num}: {len(new)} new variants (total {len(all_variants)})")
        if not current_gen:
            break

    return all_variants


def fetch_growmax_compounds(seed_smiles: str, max_variants: int = 200, rounds: int = 2):
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

    rounds = max(1, min(int(rounds), 3))

    try:
        variants = generate_variants(seed_smiles, max_variants, rounds)
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
            "Round": c["name"].split("_r")[-1] if "_r" in c["name"] else "1",
            "SMILES": c["smiles"][:60] + ("..." if len(c["smiles"]) > 60 else ""),
        }
        for c in compounds
    ]

    status = (
        f"**Step 2 – GrowMax:** ✅ Generated {len(compounds)} variants "
        f"({rounds} round{'s' if rounds > 1 else ''} of growth)"
    )
    logger.info(f"=== STEP 2 (GrowMax) COMPLETE: {len(compounds)} variants, {rounds} rounds ===")
    return status, compounds, pd.DataFrame(rows)
