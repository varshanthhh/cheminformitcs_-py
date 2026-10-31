from collections import Counter
import base64
from pathlib import Path

import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors

try:
    from rdkit.Chem import Draw
    DRAW_AVAILABLE = True
except (ImportError, OSError):
    Draw = None
    DRAW_AVAILABLE = False


ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "molecule_library.csv"


def svg_data_uri(path):
    """Embed a project SVG safely in Streamlit's custom HTML blocks."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


MASCOT_LOGO = svg_data_uri(ROOT / "assets" / "chematlas_mascot.svg")
ATOM_LOGO = svg_data_uri(ROOT / "assets" / "atom_companion.svg")

st.set_page_config(
    page_title="ChemAtlas | Chemistry Workspace",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #162826;
        --muted: #64736f;
        --paper: #f4f6f1;
        --card: #ffffff;
        --green: #174b3a;
        --lime: #d9ef8b;
        --line: #dfe6de;
    }
    [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(244, 246, 241, .9); }
    [data-testid="stSidebar"] { background: #152a28; }
    [data-testid="stSidebar"] * { color: #edf4e8; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius: 9px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #24423a; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -.025em; }
    .eyebrow { color: #4f765d; font-size: .76rem; font-weight: 800; letter-spacing: .17em; text-transform: uppercase; }
    .hero {
        background: linear-gradient(120deg, #173b33 0%, #245a43 68%, #64844b 100%);
        color: #f7f8ef; border-radius: 22px; padding: 2.3rem 2.5rem;
        margin: .5rem 0 1.5rem 0; box-shadow: 0 15px 40px rgba(18, 53, 42, .12);
        position: relative; overflow: hidden; min-height: 255px;
    }
    .hero h1 { color: #fbfff1; font-size: clamp(2.3rem, 5vw, 4.2rem); margin: .4rem 0; }
    .hero p { color: #dce9d7; font-size: 1.05rem; max-width: 650px; }
    .hero-tag { color: #d9ef8b; font-weight: 800; letter-spacing: .14em; font-size: .76rem; }
    .hero-copy { position: relative; z-index: 2; max-width: 72%; }
    .hero-mascot { position: absolute; width: 230px; right: 3%; bottom: -18px; z-index: 1; filter: drop-shadow(0 12px 15px rgba(8, 25, 19, .22)); }
    .sidebar-brand { display: flex; align-items: center; gap: .75rem; margin: .2rem 0 .4rem; }
    .sidebar-brand img { width: 54px; height: 54px; border-radius: 16px; background: #d9ef8b; }
    .sidebar-brand strong { display: block; color: #f8ffef; font-size: 1.1rem; }
    .sidebar-brand span { display: block; color: #bed1c4; font-size: .72rem; letter-spacing: .11em; text-transform: uppercase; }
    .panel {
        background: var(--card); border: 1px solid var(--line); border-radius: 16px;
        padding: 1rem 1.15rem; margin: .25rem 0 .75rem 0;
    }
    .tiny-note { color: var(--muted); font-size: .9rem; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px; border: 1px solid #174b3a; font-weight: 700;
    }
    .stButton > button[kind="primary"] { background: #174b3a; }
    code { color: #174b3a !important; }
    @media (max-width: 720px) {
        .hero { padding: 1.7rem 1.3rem; }
        .hero-copy { max-width: 100%; }
        .hero-mascot { width: 115px; opacity: .24; right: 1%; bottom: -4px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_library():
    return pd.read_csv(DATA_FILE).fillna("")


LIBRARY = load_library()
MOLECULES = LIBRARY.set_index("Molecule").to_dict(orient="index")


def parse_smiles(smiles):
    if not smiles or not str(smiles).strip():
        return None
    try:
        return Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None


def molecular_properties(mol):
    return {
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight (g/mol)": round(Descriptors.MolWt(mol), 3),
        "Exact mass (Da)": round(Descriptors.ExactMolWt(mol), 4),
        "Calculated LogP": round(Crippen.MolLogP(mol), 3),
        "Topological polar surface area (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "H-bond donors": Lipinski.NumHDonors(mol),
        "H-bond acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "Heavy atoms": mol.GetNumHeavyAtoms(),
        "Formal charge": Chem.GetFormalCharge(mol),
    }


def property_row(name, smiles):
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    return {"Molecule": name, "SMILES": Chem.MolToSmiles(mol), **molecular_properties(mol)}


def atom_and_charge_counts(mol):
    elements = Counter()
    charge = 0
    for atom in mol.GetAtoms():
        elements[atom.GetSymbol()] += 1
        elements["H"] += atom.GetTotalNumHs(includeNeighbors=False)
        charge += atom.GetFormalCharge()
    return elements, charge


def parse_reaction(reaction_text):
    text = reaction_text.strip()
    if ">>" in text:
        pieces = text.split(">>")
        if len(pieces) != 2:
            return None, None, "Use one reaction arrow, written as reactants>>products."
        reactant_text, product_text = pieces
    elif ">" in text:
        pieces = text.split(">")
        if len(pieces) != 3:
            return None, None, "Use reactants>reagents>products, or reactants>>products."
        reactant_text, product_text = pieces[0], pieces[2]
    else:
        return None, None, "Add a reaction arrow: reactants>>products."

    if not reactant_text.strip() or not product_text.strip():
        return None, None, "Add at least one reactant and one product."

    def side_counts(side):
        total_elements = Counter()
        total_charge = 0
        components_seen = []
        for component in side.split("."):
            component = component.strip()
            if not component:
                continue
            mol = parse_smiles(component)
            if mol is None:
                return None, None, component
            elements, charge = atom_and_charge_counts(mol)
            total_elements.update(elements)
            total_charge += charge
            components_seen.append(Chem.MolToSmiles(mol))
        if not components_seen:
            return None, None, "empty side"
        return total_elements, total_charge, None

    reactants, reactant_charge, invalid = side_counts(reactant_text)
    if invalid:
        return None, None, f"Could not parse reactant component: {invalid}"
    products, product_charge, invalid = side_counts(product_text)
    if invalid:
        return None, None, f"Could not parse product component: {invalid}"

    elements = sorted(set(reactants) | set(products))
    rows = [
        {
            "Element": element,
            "Reactants": reactants[element],
            "Products": products[element],
            "Difference": reactants[element] - products[element],
            "Balanced": reactants[element] == products[element],
        }
        for element in elements
    ]
    rows.append(
        {
            "Element": "Net formal charge",
            "Reactants": reactant_charge,
            "Products": product_charge,
            "Difference": reactant_charge - product_charge,
            "Balanced": reactant_charge == product_charge,
        }
    )
    return pd.DataFrame(rows), all(row["Balanced"] for row in rows), None


def render_3d(smiles):
    mol = parse_smiles(smiles)
    if mol is None:
        st.error("Enter a valid SMILES string before generating a 3D conformer.")
        return
    mol3d = Chem.AddHs(Chem.Mol(mol))
    status = AllChem.EmbedMolecule(mol3d, AllChem.ETKDGv3())
    if status != 0:
        st.warning("RDKit could not generate a conformer for this structure.")
        return
    try:
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=150)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol3d, maxIters=150)
    view = py3Dmol.view(width=720, height=430)
    view.addModel(Chem.MolToMolBlock(mol3d), "sdf")
    view.setStyle({"stick": {"radius": 0.16}, "sphere": {"scale": 0.24}})
    view.setBackgroundColor("#fbfcf8")
    view.zoomTo()
    components.html(view._make_html(), height=450)


def page_heading(kicker, title, description):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.write(description)


def navigate_to(destination):
    st.session_state["sidebar_radio"] = destination


with st.sidebar:
    st.markdown(
        f'<div class="sidebar-brand"><img src="{ATOM_LOGO}" alt="ChemAtlas anime atom mascot"><div><strong>ChemAtlas</strong><span>molecule club</span></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("A small workspace for chemical structure and data.")
    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Molecule studio",
            "Library & compare",
            "Reaction balance",
            "Learn",
        ],
        label_visibility="visible",
        key="sidebar_radio",
    )
    st.divider()
    st.markdown("**STRUCTURE ENGINE**")
    st.caption("SMILES parsing and descriptors powered by RDKit.")
    st.markdown("**LIBRARY**")
    st.caption(f"{len(LIBRARY)} curated structures")


if page == "Overview":
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-copy">
            <div class="hero-tag">CHEMISTRY · DATA · DISCOVERY</div>
            <h1>Explore the language<br>of molecules.</h1>
            <p>Turn a SMILES string into a structure, a set of interpretable descriptors,
            or a quick comparison. Then check whether a reaction conserves its atoms.</p>
          </div>
          <img class="hero-mascot" src="{MASCOT_LOGO}" alt="Original anime-inspired ChemAtlas lab mascot">
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Reference molecules", len(LIBRARY), "expanded sample set")
    metric_2.metric("Structure tools", "4", "explore · compare · balance · learn")
    metric_3.metric("Input format", "SMILES", "canonicalized with RDKit")

    st.markdown("### Pick a starting point")
    cards = st.columns(3)
    with cards[0]:
        st.markdown('<div class="panel"><b>01 · Molecule studio</b><br><span class="tiny-note">Draw and inspect one structure, then review its calculated properties.</span></div>', unsafe_allow_html=True)
        st.button("Open molecule studio", use_container_width=True, on_click=navigate_to, args=("Molecule studio",))
    with cards[1]:
        st.markdown('<div class="panel"><b>02 · Library & compare</b><br><span class="tiny-note">Browse the refreshed compound set and compare descriptor profiles.</span></div>', unsafe_allow_html=True)
        st.button("Browse the library", use_container_width=True, on_click=navigate_to, args=("Library & compare",))
    with cards[2]:
        st.markdown('<div class="panel"><b>03 · Reaction balance · NEW</b><br><span class="tiny-note">Check element counts and net formal charge across a reaction SMILES.</span></div>', unsafe_allow_html=True)
        st.button("Check a reaction", use_container_width=True, on_click=navigate_to, args=("Reaction balance",))

    st.markdown("### A different molecule set")
    st.dataframe(
        LIBRARY[["Molecule", "Family", "SMILES", "Context"]],
        use_container_width=True,
        hide_index=True,
        height=360,
    )
    st.caption("Descriptor values are calculated from each structure at runtime. The library is a learning set, not an experimental database.")

elif page == "Molecule studio":
    page_heading(
        "WORKBENCH 01",
        "Molecule studio",
        "Explore a library structure or paste your own SMILES to see its 2D form and calculated descriptors.",
    )
    sample_col, load_col = st.columns([3, 1])
    with sample_col:
        selected_sample = st.selectbox("Library sample", ["Custom structure", *LIBRARY["Molecule"].tolist()])
    with load_col:
        st.write("")
        st.write("")
        load_clicked = st.button("Load sample", use_container_width=True)
    if load_clicked and selected_sample != "Custom structure":
        st.session_state["studio_smiles"] = MOLECULES[selected_sample]["SMILES"]
    smiles = st.text_input(
        "SMILES",
        value=st.session_state.get("studio_smiles", "CCO"),
        key="studio_smiles",
        help="Examples: CCO for ethanol, c1ccccc1 for benzene.",
    )
    molecule = parse_smiles(smiles)
    if molecule is None:
        st.error("That SMILES string could not be parsed. Check the atom symbols, branches, and ring numbers.")
    else:
        properties = molecular_properties(molecule)
        structure_col, data_col = st.columns([1, 1.25], gap="large")
        with structure_col:
            st.markdown("#### 2D structure")
            if DRAW_AVAILABLE:
                st.image(Draw.MolToImage(molecule, size=(620, 430)), use_container_width=True)
            else:
                st.info("2D structure rendering is unavailable in this runtime; the SMILES and calculated profile are still shown.")
            st.code(Chem.MolToSmiles(molecule), language=None)
            st.caption("Canonical SMILES generated by RDKit.")
        with data_col:
            st.markdown("#### Calculated profile")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Molecular weight", f"{properties['Molecular weight (g/mol)']:.2f} g/mol")
            metric_cols[1].metric("Calculated LogP", f"{properties['Calculated LogP']:.2f}")
            metric_cols[2].metric("TPSA", f"{properties['Topological polar surface area (Å²)']:.1f} Å²")
            property_df = pd.DataFrame(
                [{"Property": name, "Value": str(value)} for name, value in properties.items()]
            )
            st.dataframe(property_df, use_container_width=True, hide_index=True)
            csv = pd.DataFrame([{"Input SMILES": smiles, "Canonical SMILES": Chem.MolToSmiles(molecule), **properties}]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download molecule report (CSV)",
                csv,
                file_name="chematlas_molecule_report.csv",
                mime="text/csv",
            )
        with st.expander("Generate an interactive 3D conformer"):
            st.caption("This is a computed conformer for exploration. It is not an experimentally determined geometry.")
            if st.button("Generate 3D view", key="generate_3d"):
                render_3d(smiles)

elif page == "Library & compare":
    page_heading(
        "WORKBENCH 02",
        "Library & compare",
        "Browse a refreshed set of everyday, biological, and pharmaceutical structures, then compare their descriptors.",
    )
    search_col, family_col = st.columns([2, 1])
    with search_col:
        query = st.text_input("Search molecules, SMILES, or context", placeholder="Try: aromatic, acid, caffeine")
    families = ["All families", *sorted(LIBRARY["Family"].unique())]
    with family_col:
        family = st.selectbox("Family", families)
    filtered = LIBRARY.copy()
    if family != "All families":
        filtered = filtered[filtered["Family"] == family]
    if query.strip():
        needle = query.strip().lower()
        mask = filtered.astype(str).apply(
            lambda column: column.str.lower().str.contains(needle, regex=False)
        ).any(axis=1)
        filtered = filtered[mask]
    st.dataframe(filtered, use_container_width=True, hide_index=True, height=310)
    st.download_button(
        "Download current library view (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="chematlas_molecule_library.csv",
        mime="text/csv",
    )

    st.markdown("### Compare descriptor profiles")
    choices = st.multiselect(
        "Choose two or more molecules",
        options=LIBRARY["Molecule"].tolist(),
        default=LIBRARY["Molecule"].head(3).tolist(),
    )
    if len(choices) < 2:
        st.info("Choose at least two molecules to compare.")
    else:
        compared = pd.DataFrame(
            [property_row(name, MOLECULES[name]["SMILES"]) for name in choices]
        )
        st.dataframe(compared, use_container_width=True, hide_index=True)
        st.markdown("#### Molecular weight")
        st.bar_chart(compared.set_index("Molecule")[["Molecular weight (g/mol)"]], color="#34785a")
        st.download_button(
            "Download comparison (CSV)",
            compared.to_csv(index=False).encode("utf-8"),
            file_name="chematlas_comparison.csv",
            mime="text/csv",
        )

elif page == "Reaction balance":
    page_heading(
        "WORKBENCH 03 · NEW TOOL",
        "Reaction atom balance",
        "Check whether a reaction preserves each element and its total formal charge using reaction SMILES.",
    )
    st.markdown(
        '<div class="panel"><b>Format</b><br><span class="tiny-note">Write reactants and products as dot-separated SMILES with a double arrow: reactants&gt;&gt;products. A three-part form, reactants&gt;reagents&gt;products, is also accepted. Repeat components to represent stoichiometric coefficients.</span></div>',
        unsafe_allow_html=True,
    )
    reaction_text = st.text_area(
        "Reaction SMILES",
        value="CCO.O=O.O=O.O=O>>O=C=O.O=C=O.O.O.O",
        height=100,
        help="Example: ethanol combustion, with repeated oxygen and water components.",
    )
    if st.button("Check atom balance", type="primary"):
        balance_table, is_balanced, error = parse_reaction(reaction_text)
        if error:
            st.error(error)
        elif is_balanced:
            st.success("Balanced: element counts and net formal charge match.")
            st.dataframe(balance_table, use_container_width=True, hide_index=True)
        else:
            st.warning("Not balanced: at least one element count or the net formal charge differs.")
            st.dataframe(balance_table, use_container_width=True, hide_index=True)
            csv = balance_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download balance check (CSV)",
                csv,
                file_name="reaction_balance.csv",
                mime="text/csv",
            )
    st.caption("This check tests conservation only. It does not predict products, validate a mechanism, or assess reaction conditions or yield.")

elif page == "Learn":
    page_heading(
        "FIELD NOTES",
        "A short guide to the structures",
        "A few useful ideas for reading the molecular data in this workspace.",
    )
    with st.expander("SMILES is a line notation"):
        st.write("SMILES encodes atoms and connections as text. Parentheses mark branches, digits close rings, and lower-case aromatic symbols represent aromatic atoms.")
        st.code("CC(=O)O   acetic acid\nc1ccccc1   benzene\nCCO        ethanol", language=None)
    with st.expander("Descriptors summarize different features"):
        st.write("Molecular weight summarizes mass. Calculated LogP is a model-based lipophilicity estimate. TPSA summarizes polar surface area. H-bond donor and acceptor counts describe selected interaction features.")
        st.info("Descriptors are calculated estimates. A single number does not determine a molecule's measured behavior.")
    with st.expander("Atom balance is a conservation check"):
        st.write("A balanced reaction has the same number of every element on both sides. For ionic structures, the net formal charge should also be conserved.")
        st.code("CCO.O=O.O=O.O=O>>O=C=O.O=C=O.O.O.O", language=None)
    st.markdown("### Quick check")
    answer = st.radio(
        "In a SMILES string, what does a matching pair of digits usually indicate?",
        ["A ring closure", "A formal charge", "A molecular weight"],
        index=None,
        key="learn_quiz",
    )
    if st.button("Check answer"):
        if answer == "A ring closure":
            st.success("Correct. Matching digits mark the two atoms connected to close a ring.")
        elif answer is None:
            st.warning("Choose an answer first.")
        else:
            st.info("Not quite. In SMILES, matching digits indicate a ring closure.")


