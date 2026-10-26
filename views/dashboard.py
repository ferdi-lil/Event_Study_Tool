import pandas as pd
import streamlit as st

from eventstudy import ai, storage
from eventstudy.config import StudyConfig
from ui import style
from ui.state import open_results, start_new

studies = storage.list_studies()
style.titlebar("Studien", f"{len(studies)} Studien · KI-Hilfe {'aktiv' if ai.available() else 'inaktiv'}")

c1, c2, _ = st.columns([1.2, 1.2, 4])
if c1.button("Neue Studie", type="primary", use_container_width=True):
    start_new()
    st.switch_page("views/wizard.py")
with c2.popover("Aus Vorlage", use_container_width=True):
    for stem, tpl in storage.list_templates():
        st.markdown(f"**{tpl.name}**")
        st.caption(tpl.description[:220] + ("…" if len(tpl.description) > 220 else ""))
        if st.button("Vorlage öffnen", key=f"tpl_{stem}"):
            start_new(StudyConfig.from_dict(tpl.to_dict()))
            st.switch_page("views/wizard.py")

style.section("Studien")
if not studies:
    st.markdown('<div class="es-note">Noch keine Studien vorhanden. „Neue Studie“ anlegen oder die Vorlage zur '
                'AI-Act-Bachelorarbeit öffnen.</div>', unsafe_allow_html=True)

for s in studies:
    cfg: StudyConfig = s["config"]
    done = s["status"] == "Ausgewertet"
    with st.container(border=True):
        a, b, c = st.columns([4.2, 2.6, 1.6], vertical_alignment="top")
        events = ", ".join(f"{pd.Timestamp(e.date):%d.%m.%Y} {e.label}".strip() for e in cfg.events) or "–"
        treat = ", ".join(f"{g.name} ({len(g.tickers)})" for g in cfg.treatment_groups()) or "–"
        ctrl = ", ".join(f"{g.name} ({len(g.tickers)})" for g in cfg.control_groups()) or "–"
        a.markdown(
            f'<div class="es-row-name">{cfg.name}</div>'
            f'<div class="es-row-meta">{style.tag("Ausgewertet", "ok") if done else style.tag("Entwurf", "draft")} '
            f'&nbsp;geändert {s["updated"]:%d.%m.%Y %H:%M}<br>'
            f'Events: {events}<br>Untersuchung: {treat}<br>Kontrolle: {ctrl}</div>', unsafe_allow_html=True)
        if done:
            res = storage.load_result(s["slug"])
            if res is not None and not res.summary.empty:
                main = res.main_table()
                w = StudyConfig.window_label(cfg.windows[0])
                main = main[(main["Role"] == "treatment") & (main["Window"] == w)].head(2)
                with b:
                    style.kpis([(f"{r['Group']} {w} · {pd.Timestamp(r['Event']):%d.%m.%y}",
                                 f"{r['CAAR'] * 100:+.2f} %", f"p = {r.get('p_t', float('nan')):.3f} (t-Test)",
                                 0) for _, r in main.iterrows()])
        with c:
            if done and st.button("Ergebnisse", key=f"open_{s['slug']}", use_container_width=True):
                open_results(s["slug"])
                st.switch_page("views/results.py")
            if st.button("Bearbeiten", key=f"edit_{s['slug']}", use_container_width=True):
                start_new(cfg, s["slug"])
                st.switch_page("views/wizard.py")
            with st.popover("Weitere", use_container_width=True):
                if st.button("Duplizieren", key=f"dup_{s['slug']}"):
                    copy = StudyConfig.from_dict(cfg.to_dict())
                    copy.name = cfg.name + " (Kopie)"
                    storage.save_config(copy)
                    st.rerun()
                if st.button("Löschen", key=f"del_{s['slug']}"):
                    storage.delete_study(s["slug"])
                    st.rerun()

st.markdown('<div class="es-note" style="margin-top:18px">Datenquellen: Yahoo Finance (Kurse), Kenneth R. French '
            'Data Library (Faktoren) oder eigene Datenpakete.</div>', unsafe_allow_html=True)
