import gradio as gr

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

from tabs.pipeline.detail import draw_2d_with_pharmacophore


# Canonical goal labels shown to the user (checkboxes)
OPTIMIZATION_GOAL_OPTIONS = [
    "Increase solubility",
    "Improve aqueous solubility",
    "Reduce lipophilicity",
    "Optimize logP",
    "Increase lipophilicity",
    "Improve BBB permeability",
    "Improve metabolic stability",
    "Reduce toxicity",
    "Increase polarity",
    "Increase permeability",
]


def _optimize_compound(smiles: str, goals: list[str]):
    """
    Compound Optimization Tab Logic

    Inputs:
      - smiles: SMILES string for the compound
      - goals: list of selected optimization goals (checkbox selections)

    Outputs:
      - formatted text summary of properties + goal-based suggestions + Lipinski assessment
      - 2D molecule image rendered from SMILES
    """
    smiles = (smiles or "").strip()
    goals = goals or []

    if not smiles:
        return "Please enter a valid SMILES string.", None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES string. Please check your input.", None

    # --- Basic properties ---
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    # --- Lipinski rule-of-five violations ---
    violations = 0
    violations += 1 if mw > 500 else 0
    violations += 1 if logp > 5 else 0
    violations += 1 if hbd > 5 else 0
    violations += 1 if hba > 10 else 0

    # A common quick heuristic: <= 1 violation is typically acceptable in early discovery
    lipinski_status = "Pass" if violations <= 1 else "Fail"

    # --- Goal-based suggestions ---
    # Normalize selected goals for safe matching
    selected = {g.strip().lower() for g in goals if isinstance(g, str) and g.strip()}

    suggestions: list[str] = []
    goal_mapping: list[str] = []

    # Solubility / polarity cluster
    if (
        "increase solubility" in selected
        or "improve aqueous solubility" in selected
        or "increase polarity" in selected
    ):
        suggestions.append(
            "Increase polarity by adding or substituting polar functional groups such as hydroxyl (-OH), amines (-NH2), or heteroatoms in rings."
        )
        suggestions.append(
            "Reduce crystallinity by introducing small branching or disrupting symmetry, if feasible for the scaffold."
        )
        goal_mapping.append("Solubility → increase polarity, reduce crystallinity, balance LogP")

    # Lipophilicity / logP cluster
    if ("reduce lipophilicity" in selected) or ("optimize logp" in selected):
        suggestions.append(
            "Reduce lipophilicity by shortening alkyl chains, reducing aromatic ring count, or adding polar substituents while preserving key pharmacophore features."
        )
        suggestions.append(
            "Avoid adding large hydrophobic groups that increase LogP unless BBB penetration is a primary requirement."
        )
        goal_mapping.append("Lipophilicity/logP → reduce hydrophobic surface area and aromatic burden")

    # If user explicitly wants higher lipophilicity
    if "increase lipophilicity" in selected:
        suggestions.append(
            "To increase lipophilicity, consider adding small hydrophobic substituents (e.g., methyl/ethyl) or increasing aromatic character, while monitoring solubility and clearance."
        )
        suggestions.append(
            "Avoid overcorrecting. Excessive lipophilicity can increase off-target binding and reduce aqueous solubility."
        )
        goal_mapping.append("Increase lipophilicity → add hydrophobic substituents carefully, monitor solubility")

    # BBB cluster
    if "improve bbb permeability" in selected:
        suggestions.append(
            "For BBB permeability, aim for moderate lipophilicity. A common early heuristic is LogP roughly in the 1–3 range while keeping polarity controlled."
        )
        suggestions.append(
            "Minimize strongly ionizable groups when possible and avoid excessive hydrogen-bonding that can reduce CNS exposure."
        )
        goal_mapping.append("BBB permeability → moderate LogP, controlled polarity, fewer strong ionizable groups")

    # Metabolic stability cluster
    if "improve metabolic stability" in selected:
        suggestions.append(
            "To improve metabolic stability, consider reducing easily oxidized motifs (e.g., exposed aromatic methyls) and consider steric shielding near labile positions."
        )
        suggestions.append(
            "Avoid reactive functional groups and consider bioisosteres for metabolically labile linkers."
        )
        goal_mapping.append("Metabolic stability → reduce labile motifs, consider bioisosteres")

    # Toxicity / off-target cluster
    if "reduce toxicity" in selected:
        suggestions.append(
            "To reduce off-target risk, avoid high lipophilicity and consider removing structural alerts if present. Keep the molecule as simple as possible while maintaining activity."
        )
        goal_mapping.append("Safety/off-target → avoid excessive lipophilicity, reduce structural alerts")

    # Permeability (non-BBB) cluster
    if "increase permeability" in selected:
        suggestions.append(
            "To increase permeability, reduce excessive polarity and hydrogen-bond donors where feasible, and consider neutral bioisosteres for strongly ionizable groups."
        )
        suggestions.append(
            "Balance permeability with solubility. Small changes can shift both properties significantly."
        )
        goal_mapping.append("Permeability → reduce excess polarity/HBD, balance with solubility")

    if not suggestions:
        suggestions.append(
            "Select one or more optimization goals to receive tailored recommendations."
        )
        goal_mapping.append("No selected goals → choose goals for targeted suggestions")

    suggestions_text = "\n".join([f"- {s}" for s in suggestions])
    goal_mapping_text = "\n".join([f"- {m}" for m in goal_mapping])

    # --- Drug-likeness notes (simple, practical) ---
    drug_likeness_notes = [
        "Lipinski Rule of Five is a quick screen, not a guarantee of success.",
        "Optimization involves tradeoffs. Improving BBB permeability can reduce solubility.",
        "Use docking and ADME tools together to guide iteration rather than relying on one metric.",
    ]
    drug_likeness_text = "\n".join([f"- {n}" for n in drug_likeness_notes])

    # --- Build response ---
    goals_display = ", ".join(goals) if goals else "(none selected)"
    response = f"""Initial Properties:
- Molecular Weight: {mw:.2f}
- LogP: {logp:.2f}
- H-Bond Donors: {hbd}
- H-Bond Acceptors: {hba}

Optimization Goals:
{goals_display}

Suggested Structural Modifications:
{suggestions_text}

Optimization Goal Mapping:
{goal_mapping_text}

Drug-Likeness Considerations:
- Lipinski Violations: {violations}
- Overall Lipinski Status: {lipinski_status}
{drug_likeness_text}
"""

    # --- 2D image with pharmacophore highlights ---
    img = draw_2d_with_pharmacophore(mol, size=(350, 350))

    return response.strip(), img


def create_tab():
    with gr.Tab("Compound Optimization"):
        with gr.Row():
            with gr.Column(scale=1):
                optim_smiles_input = gr.Textbox(
                    label="Compound SMILES",
                    placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O",
                    lines=3,
                )

                # Checkbox selection for goals (replaces free-text)
                optim_goals_input = gr.CheckboxGroup(
                    choices=OPTIMIZATION_GOAL_OPTIONS,
                    label="Optimization Goals",
                    info="Select one or more goals.",
                )

                optimize_btn = gr.Button(
                    "Optimize Compound",
                    size="lg",
                    elem_classes=["analyze-btn"],
                )

            with gr.Column(scale=1):
                optim_output = gr.Textbox(
                    label="Optimization Suggestions",
                    interactive=False,
                    lines=14,
                    placeholder=(
                        "a. Initial Properties of the Compound\n"
                        "b. Suggested Structural Modifications\n"
                        "c. Optimization Goal Mapping\n"
                        "d. Drug-Likeness Considerations"
                    ),
                )
                optim_image = gr.Image(
                    label="Molecule Visualization (2D)",
                    type="pil",
                    height=350,
                )

        gr.Examples(
            examples=[
                ["CC(=O)Oc1ccccc1C(=O)O", "Increase solubility"],
                ["c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34", "Reduce lipophilicity"],
                ["CC(C)Cc1ccc(C(C)C(=O)O)cc1", "Improve bioavailability"],
            ],
            inputs=[optim_smiles_input, optim_goals_input],
        )

        optimize_btn.click(
            _optimize_compound,
            inputs=[optim_smiles_input, optim_goals_input],
            outputs=[optim_output, optim_image],
        )