# Deploy ChemAtlas

## Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app**.
3. Select repository `varshanthhh/cheminformitcs_-py`.
4. Select branch `chematlas-redesign`.
5. Set the main file path to `app.py`.
6. Choose **Deploy** and wait for the build to finish.

Streamlit will provide the public app URL after deployment succeeds. This branch contains the redesigned ChemAtlas app; the original `main` branch has not been changed.

## Run locally

Use Python 3.10 or newer, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app uses RDKit for SMILES parsing, 2D structure rendering, and molecular descriptors. Descriptor values are computed estimates for educational exploration, not experimental measurements.
