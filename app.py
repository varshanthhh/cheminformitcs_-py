from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, rdMolDescriptors

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "molecule_library.csv"

st.set_page_config(page_title="ChemAtlas | Molecular Explorer", page_icon="⚗️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:Manrope, sans-serif}
[data-testid="stAppViewContainer"] {background:#f5f7f2;color:#172b26}
[data-testid="stSidebar"] {background:#142c26}
[data-testid="stSidebar"] * {color:#edf5e9}
h1,h2,h3 {letter-spacing:-.04em}
.hero {padding:2.2rem 2.4rem;border-radius:24px;background:linear-gradient(120deg,#153b30,#2d694d);color:#f6faef;margin:.5rem 0 1.5rem}
.hero h1 {font-size:clamp(2.5rem,5vw,4.4rem);color:#f6faef;line-height:1.03;margin:.6rem 0}
.hero p {font-size:1.05rem;color:#dcebdc;max-width:680px}
.kicker {font:500 .76rem 'DM Mono',monospace;letter-spacing:.16em;color:#d8ef91}
.panel {padding:1rem 1.2rem;background:white;border:1px solid #dce5da;border-radius:16px}
div[data-testid="stMetric"] {background:#fff;padding:1rem;border:1px solid #dce5da;border-radius:14px}
.stButton button,.stDownloadButton button {border-radius:10px;font-weight:700}
code {font-family:'DM Mono',monospace}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_library():
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE).fillna("")
    return pd.DataFrame([
        {"Molecule":"Ethanol","Family":"Alcohol","SMILES":"CCO","Context":"Common solvent"},
        {"Molecule":"Caffeine","Family":"Alkaloid","SMILES":"Cn1c(=O)c2c(ncn2C)n(C)c1=O","Context":"Coffee and tea"},
        {"Molecule":"Aspirin","Family":"Pharmaceutical","SMILES":"CC(=O)Oc1ccccc1C(=O)O","Context":"Analgesic"},
        {"Molecule":"Benzene","Family":"Aromatic","SMILES":"c1ccccc1","Context":"Aromatic hydrocarbon"},
        {"Molecule":"Glucose","Family":"Carbohydrate","SMILES":"C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O","Context":"Simple sugar"},
        {"Molecule":"Lactic acid","Family":"Organic acid","SMILES":"CC(O)C(=O)O","Context":"Hydroxy acid"},
        {"Molecule":"Acetic acid","Family":"Organic acid","SMILES":"CC(=O)O","Context":"Vinegar component"},
        {"Molecule":"Nicotine","Family":"Alkaloid","SMILES":"CN1CCC[C@H]1c2cccnc2","Context":"Plant alkaloid"},
        {"Molecule":"Paracetamol","Family":"Pharmaceutical","SMILES":"CC(=O)NC1=CC=C(C=C1)O","Context":"Analgesic"},
        {"Molecule":"Ibuprofen","Family":"Pharmaceutical","SMILES":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O","Context":"Anti-inflammatory"},
        {"Molecule":"Urea","Family":"Amide","SMILES":"NC(=O)N","Context":"Nitrogen compound"},
        {"Molecule":"Glycine","Family":"Amino acid","SMILES":"NCC(=O)O","Context":"Simplest amino acid"},
        {"Molecule":"Limonene","Family":"Terpene","SMILES":"CC1=CCC(CC1)C(=C)C","Context":"Citrus aroma"},
    ])

LIBRARY = load_library()
if "Molecule" not in LIBRARY.columns or "SMILES" not in LIBRARY.columns:
    st.error("The molecule CSV needs 'Molecule' and 'SMILES' columns.")
    st.stop()
LIBRARY = LIBRARY.dropna(subset=["Molecule", "SMILES"])
MOLECULES = LIBRARY.set_index("Molecule").to_dict("index")

def parse_smiles(smiles):
    if not str(smiles).strip():
        return None
    try:
        return Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None

def props(mol):
    return {
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight (g/mol)": round(Descriptors.MolWt(mol), 3),
        "Exact mass (Da)": round(Descriptors.ExactMolWt(mol), 4),
        "Calculated LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "H-bond donors": Lipinski.NumHDonors(mol),
        "H-bond acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "Heavy atoms": mol.GetNumHeavyAtoms(),
        "Formal charge": Chem.GetFormalCharge(mol),
    }

def atom_counts(mol):
    counts = Counter()
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
        counts["H"] += atom.GetTotalNumHs(includeNeighbors=False)
    return counts, Chem.GetFormalCharge(mol)

def check_reaction(text):
    if ">>" in text:
        parts = text.split(">>")
        if len(parts) != 2: return None, "Use reactants>>products with one arrow."
        left, right = parts
    elif ">" in text:
        parts = text.split(">")
        if len(parts) != 3: return None, "Use reactants>reagents>products."
        left, right = parts[0], parts[2]
    else:
        return None, "Add a reaction arrow: reactants>>products."
    def side(raw):
        total, charge = Counter(), 0
        for s in raw.split("."):
            mol = parse_smiles(s)
            if mol is None: return None, None, s
            c, q = atom_counts(mol); total.update(c); charge += q
        return total, charge, None
    a, qa, err = side(left)
    if err: return None, f"Could not parse reactant: {err}"
    b, qb, err = side(right)
    if err: return None, f"Could not parse product: {err}"
    rows = [{"Element":e,"Reactants":a[e],"Products":b[e],"Difference":a[e]-b[e],"Balanced":a[e]==b[e]} for e in sorted(set(a)|set(b))]
    rows.append({"Element":"Net formal charge","Reactants":qa,"Products":qb,"Difference":qa-qb,"Balanced":qa==qb})
    return pd.DataFrame(rows), None

with st.sidebar:
    st.markdown("## ⚗️ ChemAtlas")
    st.caption("MOLECULAR EXPLORATION LAB")
    page = st.radio("WORKSPACE", ["Overview","Molecule studio","Library & compare","Reaction checker","Learn"])
    st.divider()
    st.caption(f"{len(LIBRARY)} reference structures")
    st.caption("Powered by RDKit")

if page == "Overview":
    st.markdown('<div class="hero"><div class="kicker">CHEMISTRY · DATA · DISCOVERY</div><h1>Explore the language<br>of molecules.</h1><p>Turn chemical notation into structures, interpretable descriptors, and side-by-side comparisons. Check reaction atom balance in seconds.</p></div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.metric("Reference molecules",len(LIBRARY))
    b.metric("Workspace tools","04")
    c.metric("Structure input","SMILES")
    st.subheader("Choose your workbench")
    c1,c2,c3 = st.columns(3)
    c1.markdown("### 🧬 Molecule studio\nInspect a structure and calculate molecular properties.")
    c2.markdown("### 📊 Library & compare\nSearch compounds and compare descriptor profiles.")
    c3.markdown("### ⚖️ Reaction checker\nCheck element counts and formal charge.")
    st.subheader("Reference library")
    st.dataframe(LIBRARY,use_container_width=True,hide_index=True)
    st.caption("Educational reference set. Calculated descriptors are estimates, not experimental measurements.")

elif page == "Molecule studio":
    st.title("Molecule studio")
    st.write("Select a reference compound or paste a SMILES string.")
    sample = st.selectbox("Reference molecule",["Custom structure",*LIBRARY["Molecule"].astype(str).tolist()])
    default_smiles = "CCO" if sample == "Custom structure" else str(MOLECULES[sample]["SMILES"])
    smiles = st.text_input("SMILES",value=default_smiles,help="Examples: CCO (ethanol), c1ccccc1 (benzene)")
    mol = parse_smiles(smiles)
    if mol is None:
        st.error("Could not parse this SMILES. Check atom symbols, branches, and ring numbers.")
    else:
        p = props(mol)
        left,right = st.columns([1,1.2],gap="large")
        with left:
            st.subheader("2D structure")
            st.image(Draw.MolToImage(mol,size=(560,420)),use_container_width=True)
            st.caption("Canonical SMILES")
            st.code(Chem.MolToSmiles(mol))
        with right:
            st.subheader("Calculated profile")
            m1,m2,m3 = st.columns(3)
            m1.metric("Mol. weight",f'{p["Molecular weight (g/mol)"]:.2f}')
            m2.metric("LogP",p["Calculated LogP"])
            m3.metric("TPSA",p["TPSA (Å²)"])
            st.dataframe(pd.DataFrame([{"Property":k,"Value":v} for k,v in p.items()]),hide_index=True,use_container_width=True)
            report = pd.DataFrame([{"Input SMILES":smiles,"Canonical SMILES":Chem.MolToSmiles(mol),**p}])
            st.download_button("⬇ Download molecule report",report.to_csv(index=False).encode(),file_name="chematlas_molecule_report.csv",mime="text/csv",use_container_width=True)

elif page == "Library & compare":
    st.title("Library & compare")
    query = st.text_input("Search compounds, SMILES, family, or context",placeholder="Try caffeine, aromatic, acid…")
    families = ["All families"] + sorted(LIBRARY.get("Family",pd.Series(["Unclassified"])).astype(str).unique().tolist())
    family = st.selectbox("Molecule family",families)
    view = LIBRARY.copy()
    if family != "All families" and "Family" in view: view = view[view["Family"].astype(str)==family]
    if query.strip():
        mask = view.astype(str).apply(lambda col: col.str.contains(query.strip(),case=False,regex=False)).any(axis=1)
        view = view[mask]
    st.dataframe(view,use_container_width=True,hide_index=True)
    st.download_button("⬇ Download filtered library",view.to_csv(index=False).encode(),file_name="chematlas_library.csv",mime="text/csv")
    st.divider()
    st.subheader("Compare descriptors")
    selected = st.multiselect("Choose at least two molecules",LIBRARY["Molecule"].astype(str).tolist(),default=LIBRARY["Molecule"].astype(str).head(min(3,len(LIBRARY))).tolist())
    rows=[]
    for name in selected:
        mol=parse_smiles(str(MOLECULES[name]["SMILES"]))
        if mol: rows.append({"Molecule":name,**props(mol)})
    if len(rows)>=2:
        df=pd.DataFrame(rows)
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.bar_chart(df.set_index("Molecule")[["Molecular weight (g/mol)"]])
        st.download_button("⬇ Download comparison",df.to_csv(index=False).encode(),file_name="chematlas_comparison.csv",mime="text/csv")
    else:
        st.info("Choose two or more valid structures to compare.")

elif page == "Reaction checker":
    st.title("Reaction atom balance")
    st.write("Check whether each element and the total formal charge are conserved. Enter reaction SMILES; repeat components to represent coefficients.")
    reaction = st.text_area("Reaction SMILES",value="CCO.O=O.O=O.O=O>>O=C=O.O=C=O.O.O.O",height=100)
    if st.button("Check atom balance",type="primary"):
        result,error=check_reaction(reaction.strip())
        if error: st.error(error)
        elif result["Balanced"].all(): st.success("Balanced: element counts and net formal charge match."); st.dataframe(result,use_container_width=True,hide_index=True)
        else:
            st.warning("Not balanced: at least one element count or net formal charge differs.")
            st.dataframe(result,use_container_width=True,hide_index=True)
            st.download_button("Download balance report",result.to_csv(index=False).encode(),file_name="reaction_balance.csv",mime="text/csv")
    st.caption("This is a conservation check, not a reaction predictor. It does not evaluate mechanisms, conditions, or yields.")

else:
    st.title("Learn the language of molecules")
    with st.expander("SMILES in a minute",expanded=True):
        st.write("SMILES represents atoms and bonds as text. Parentheses describe branches, matching digits close rings, and lowercase symbols often represent aromatic atoms.")
        st.code("CCO        ethanol\nc1ccccc1  benzene\nCC(=O)O   acetic acid")
    with st.expander("What do descriptors mean?"):
        st.write("Molecular weight estimates mass; calculated LogP estimates lipophilicity; TPSA summarizes polar surface area; donor and acceptor counts describe selected hydrogen-bonding features.")
    with st.expander("What does atom balance check?"):
        st.write("A balanced reaction has the same count of each element on both sides. For ionic structures, the net formal charge should also match.")
    st.subheader("Quick check")
    answer=st.radio("What do matching digits usually indicate in SMILES?",["A ring closure","Molecular weight","A formal charge"],index=None)
    if st.button("Check answer"):
        if answer=="A ring closure": st.success("Correct — matching digits mark atoms connected to close a ring.")
        elif answer: st.info("Not quite. Matching digits usually indicate a ring closure.")
