"""Event-Study-Tool – start with:  streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from ui import style  # noqa: E402

st.set_page_config(page_title="Event Study Workbench", layout="wide")
style.apply()

pages = [
    st.Page("views/dashboard.py", title="Studien", default=True),
    st.Page("views/wizard.py", title="Studie einrichten"),
    st.Page("views/results.py", title="Ergebnisse"),
    st.Page("views/methodik.py", title="Methodik"),
]
st.navigation(pages).run()
