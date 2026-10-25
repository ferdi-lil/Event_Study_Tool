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

st.set_page_config(page_title="Event-Study-Tool", page_icon="📈", layout="wide")

pages = [
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page("views/wizard.py", title="Neue Event Study", icon=":material/add_chart:"),
    st.Page("views/results.py", title="Ergebnisse", icon=":material/insights:"),
    st.Page("views/methodik.py", title="Methodik", icon=":material/menu_book:"),
]
st.navigation(pages).run()
