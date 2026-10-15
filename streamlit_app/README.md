# Synthetic Data Lab — Streamlit interface

This directory contains the Aura-themed Streamlit UI for the project. It uses the datasets, trained CTGAN model, scripts, and reports in the repository root.

## Run on Windows

From the repository root, create and activate a virtual environment, install the root requirements, and start Streamlit:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app/app.py
```

Open the localhost URL printed by Streamlit. Local accounts are stored as salted password hashes in `.streamlit/auth.sqlite3`, which is ignored by Git. This authentication is for local demonstrations and is not production-ready.