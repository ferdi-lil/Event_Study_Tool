"""Session-state helpers shared by the pages."""
from __future__ import annotations

import streamlit as st

from eventstudy.config import StudyConfig


def draft() -> StudyConfig:
    if "draft" not in st.session_state:
        st.session_state["draft"] = StudyConfig(name="Neue Event Study")
        st.session_state["draft_slug"] = None
        st.session_state["wizard_step"] = 0
    return st.session_state["draft"]


def start_new(cfg: StudyConfig | None = None, slug: str | None = None) -> None:
    st.session_state["draft"] = cfg or StudyConfig(name="Neue Event Study")
    st.session_state["draft_slug"] = slug
    st.session_state["wizard_step"] = 0
    for k in [k for k in st.session_state if k.startswith("ge_")]:
        del st.session_state[k]


def open_results(slug: str) -> None:
    st.session_state["open_slug"] = slug
