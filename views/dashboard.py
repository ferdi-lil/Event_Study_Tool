import pandas as pd
import streamlit as st

from eventstudy import ai, storage
from eventstudy.config import StudyConfig
from ui.state import open_results, start_new

st.title("📈 Event-Study-Tool")
st.caption("Kapitalmarktreaktionen auf beliebige Ereignisse messen – von der Gruppenauswahl bis zur fertigen Auswertung.")

c1, c2, c3 = st.columns([1, 1, 2])
if c1.button("➕ Neue Event Study", type="primary", use_container_width=True):
    start_new()
    st.switch_page("views/wizard.py")
templates = storage.list_templates()
with c2.popover("📋 Aus Vorlage", use_container_width=True):
    for stem, tpl in templates:
        st.markdown(f"**{tpl.name}**")
        st.caption(tpl.description[:220] + ("…" if len(tpl.description) > 220 else ""))
        if st.button("Vorlage öffnen", key=f"tpl_{stem}"):
            start_new(StudyConfig.from_dict(tpl.to_dict()))
            st.switch_page("views/wizard.py")
c3.caption(("✨ KI-Hilfe aktiv" if ai.available() else "KI-Hilfe inaktiv – `ANTHROPIC_API_KEY` in `.env` setzen")
           + " · Daten: Yahoo Finance & Kenneth R. French Data Library")

studies = storage.list_studies()
st.subheader(f"Meine Studien ({len(studies)})")
if not studies:
    st.info("Noch keine Studien. Starte eine neue Event Study oder öffne die Vorlage zur AI-Act-Bachelorarbeit.")

for s in studies:
    cfg: StudyConfig = s["config"]
    with st.container(border=True):
        a, b = st.columns([3, 2])
        badge = "🟢" if s["status"] == "Ausgewertet" else "🟡"
        a.markdown(f"### {cfg.name}")
        a.caption(f"{badge} {s['status']} · zuletzt geändert {s['updated']:%d.%m.%Y %H:%M}")
        events = ", ".join(f"{e.date} ({e.label})" if e.label else e.date for e in cfg.events) or "–"
        groups = " · ".join(f"{'🎯' if g.role == 'treatment' else '⚖️'} {g.name} ({len(g.tickers)})" for g in cfg.groups)
        a.markdown(f"**Events:** {events}  \n**Gruppen:** {groups or '–'}  \n"
                   f"**Fenster:** {', '.join(StudyConfig.window_label(w) for w in cfg.windows)}")
        if s["status"] == "Ausgewertet":
            res = storage.load_result(s["slug"])
            if res is not None and not res.summary.empty:
                main = res.main_table()
                main = main[main["Role"] == "treatment"]
                w = StudyConfig.window_label(cfg.windows[0])
                cols = b.columns(max(1, min(len(main[main['Window'] == w]), 2)))
                for i, (_, r) in enumerate(main[main["Window"] == w].head(2).iterrows()):
                    cols[i].metric(f"{r['Group']} · {pd.Timestamp(r['Event']):%d.%m.%Y}", f"{r['CAAR'] * 100:.2f} %",
                                   help=f"CAAR im Fenster {w}, p (t-Test) = {r.get('p_t', float('nan')):.4f}")
        x, y, z = b.columns([1.3, 1.3, 0.6])
        if s["status"] == "Ausgewertet" and x.button("Ergebnisse", use_container_width=True, key=f"open_{s['slug']}"):
            open_results(s["slug"])
            st.switch_page("views/results.py")
        if y.button("Bearbeiten", use_container_width=True, key=f"edit_{s['slug']}"):
            start_new(cfg, s["slug"])
            st.switch_page("views/wizard.py")
        with z.popover("⋯"):
            if st.button("Duplizieren", key=f"dup_{s['slug']}"):
                copy = StudyConfig.from_dict(cfg.to_dict())
                copy.name = cfg.name + " (Kopie)"
                storage.save_config(copy)
                st.rerun()
            if st.button("Löschen", key=f"del_{s['slug']}", type="primary"):
                storage.delete_study(s["slug"])
                st.rerun()
