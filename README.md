# ChemAtlas

ChemAtlas is a Streamlit learning workspace for exploring molecular structures and simple cheminformatics. It includes a new 13-molecule reference library, an interactive molecule studio, descriptor comparison, and a reaction atom-balance checker.

## Run locally

1. Install Python 3.10 or newer.
2. Install packages with: pip install -r requirements.txt
3. Start the app with: streamlit run app.py

## Deploy

Create a Streamlit Community Cloud app from this repository and set app.py as the main file. The molecule data is stored in data/molecule_library.csv.

## Notes

- Structures are parsed and descriptors are calculated with RDKit.
- 3D conformers are computed for visualization and are not experimental geometries.
- The reaction tool checks element counts and net formal charge. It does not predict reactions or evaluate mechanisms, conditions, or yields.
- ChemAtlas was independently rebuilt as a learning project, inspired by the [Cheminformatics Virtual Laboratory](https://rysrzappqbqxxhmesiguqgk.streamlit.app/) developed by Dr. R. Subramanian.

