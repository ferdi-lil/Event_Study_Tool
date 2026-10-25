from datetime import date

import pandas as pd
import streamlit as st

from eventstudy import ai, storage
from eventstudy.checks import check_config
from eventstudy.config import RETURN_MODELS, TESTS, Event, StudyConfig
from eventstudy.data import FrameProvider, YahooProvider, read_wide_csv
from eventstudy.engine import run_study
from ui.group_editor import group_editor, group_list
from ui.state import draft, open_results

STEPS = ["1 · Events", "2 · Untersuchungsgruppe", "3 · Kontrollgruppen", "4 · Fenster & Modell",
         "5 · Robustheit & Tests", "6 · Daten & Start"]

cfg = draft()
step = st.session_state.get("wizard_step", 0)

st.title(cfg.name if cfg.name != "Neue Event Study" else "Neue Event Study")
nav = st.radio("Schritt", STEPS, index=step, horizontal=True, label_visibility="collapsed")
if STEPS.index(nav) != step:
    st.session_state["wizard_step"] = step = STEPS.index(nav)
    st.rerun()
st.progress((step + 1) / len(STEPS))


def nav_buttons():
    st.divider()
    a, _, b = st.columns([1, 4, 1])
    if step > 0 and a.button("← Zurück", use_container_width=True):
        st.session_state["wizard_step"] = step - 1
        st.rerun()
    if step < len(STEPS) - 1 and b.button("Weiter →", type="primary", use_container_width=True):
        st.session_state["wizard_step"] = step + 1
        st.rerun()


# ------------------------------------------------------------------- step 1
if step == 0:
    cfg.name = st.text_input("Name der Studie", cfg.name)
    cfg.description = st.text_area("Forschungsfrage / Beschreibung", cfg.description, height=90,
                                   placeholder="Wie reagieren europäische Tech-Aktien auf die Verabschiedung des EU AI Act?")
    st.subheader("Event-Daten")
    with st.expander("🧭 Leitfaden: Welche Daten eignen sich als Event?"):
        st.markdown("""
- **Überraschung zählt:** Märkte reagieren auf *neue* Information. Ein Verfahrensschritt, der seit Wochen erwartet wird, hat kaum Informationsgehalt (Binder 1985).
- **Wenige, klare Ereignisse:** Lieber 1–3 Schlüsselereignisse als jeden Zwischenschritt. Zu viele Events verwässern die Effekte und erzeugen überlappende Fenster.
- **Datum der Bekanntgabe**, nicht des Inkrafttretens. Bei Bekanntgabe nach Börsenschluss oder am Wochenende ist t=0 automatisch der nächste Handelstag.
- **Konfundierende Ereignisse prüfen:** Zinsentscheidungen, Quartalszahlen großer Konzerne, makroökonomische Daten am selben Tag.
- Mehrere Events werden **getrennt** ausgewertet, sodass du sie vergleichen kannst.
""")
    # The editor's base data must stay fixed between reruns, otherwise edits are applied twice.
    if "ge_events_base" not in st.session_state:
        st.session_state["ge_events_base"] = pd.DataFrame(
            [{"Datum": pd.Timestamp(e.date).date(), "Bezeichnung": e.label} for e in cfg.events]
            or [{"Datum": None, "Bezeichnung": ""}])
    edited = st.data_editor(st.session_state["ge_events_base"], num_rows="dynamic", use_container_width=True, hide_index=True,
                            column_config={"Datum": st.column_config.DateColumn(format="DD.MM.YYYY", max_value=date.today()),
                                           "Bezeichnung": st.column_config.TextColumn(width="large")}, key="ge_events_editor")
    cfg.events = [Event(pd.Timestamp(r["Datum"]).strftime("%Y-%m-%d"), r["Bezeichnung"] or "")
                  for _, r in edited.iterrows() if pd.notna(r["Datum"])]

    with st.expander("✨ Events mit KI finden"):
        if not ai.available():
            st.info("Für KI-Vorschläge `ANTHROPIC_API_KEY` in `.env` setzen.")
        else:
            topic = st.text_input("Thema", value=cfg.description or cfg.name, key="ai_topic")
            if st.button("Ereignisse vorschlagen"):
                with st.spinner("Claude recherchiert die Chronologie …"):
                    try:
                        st.session_state["ai_events"] = ai.suggest_events(topic)
                    except ai.AIUnavailable as exc:
                        st.error(str(exc))
            sug = st.session_state.get("ai_events")
            if sug:
                st.markdown(sug["recommendation"])
                for c in sug.get("caveats", []):
                    st.caption(f"⚠️ {c}")
                for i, e in enumerate(sug["events"]):
                    c1, c2 = st.columns([5, 1])
                    flag = "" if e["confidence"] == "sicher" else " · ⚠️ Datum unsicher – bitte prüfen"
                    c1.markdown(f"**{e['date']} – {e['label']}** (Überraschung: {e['surprise']}{flag})  \n{e['why_relevant']}")
                    if c2.button("Hinzufügen", key=f"add_ev_{i}") and e["date"] not in [x.date for x in cfg.events]:
                        cfg.events.append(Event(e["date"], e["label"]))
                        st.session_state.pop("ge_events_base", None)
                        st.session_state.pop("ge_events_editor", None)
                        st.rerun()

# ------------------------------------------------------------------- step 2
elif step == 1:
    st.subheader("Untersuchungsgruppe(n)")
    st.caption("Die Unternehmen, deren Reaktion auf das Ereignis du messen willst. Du kannst mehrere anlegen, "
               "z. B. eine enge Hauptgruppe und eine erweiterte Gruppe mit Untergruppen.")
    group_list(cfg, "treatment")
    with st.expander("➕ Untersuchungsgruppe hinzufügen", expanded=not cfg.treatment_groups()):
        group_editor(cfg, "treatment", "ge_treat")

# ------------------------------------------------------------------- step 3
elif step == 2:
    st.subheader("Kontrollgruppen")
    group_list(cfg, "control")
    if ai.available() and cfg.treatment_groups():
        with st.expander("✨ KI: Welche Kontrollgruppen ergeben Sinn?"):
            if st.button("Vorschläge holen"):
                summary = (f"{cfg.name}. {cfg.description} Events: " + "; ".join(f"{e.date} {e.label}" for e in cfg.events)
                           + ". Untersuchungsgruppen: " + "; ".join(f"{g.name}: {g.selection_note}" for g in cfg.treatment_groups()))
                with st.spinner("…"):
                    try:
                        st.session_state["ai_controls"] = ai.suggest_controls(summary)
                    except ai.AIUnavailable as exc:
                        st.error(str(exc))
            for c in st.session_state.get("ai_controls", {}).get("control_groups", []):
                st.markdown(f"**{c['name']}** – {c['logic']}  \n_Schließt aus:_ {c['what_it_rules_out']}")
            st.caption("Lege die vorgeschlagenen Gruppen anschließend unten über „Angeleitet“ oder „KI-Assistent“ an.")
    with st.expander("➕ Kontrollgruppe hinzufügen", expanded=not cfg.control_groups()):
        group_editor(cfg, "control", "ge_ctrl")

# ------------------------------------------------------------------- step 4
elif step == 3:
    st.subheader("Eventfenster")
    st.caption("Mehrere Fenster zeigen die zeitliche Struktur: Vorabreaktion (negativer Start), sofortige Reaktion, "
               "verzögerte Verarbeitung. Kurze Fenster sind weniger anfällig für Störereignisse.")
    presets = [(-1, 1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 5), (-2, 2), (-5, 5), (-10, 10), (0, 10), (0, 20), (-20, 20)]
    labels = {StudyConfig.window_label(w): w for w in presets + cfg.windows}
    chosen = st.multiselect("Fenster", list(labels), default=[StudyConfig.window_label(w) for w in cfg.windows])
    c1, c2, c3 = st.columns([1, 1, 1])
    a = c1.number_input("Eigenes Fenster: Start", -60, 60, -3)
    b = c2.number_input("Ende", -60, 60, 3)
    if c3.button("Hinzufügen", use_container_width=True) and a <= b:
        chosen.append(StudyConfig.window_label((int(a), int(b))))
        labels[StudyConfig.window_label((int(a), int(b)))] = (int(a), int(b))
    cfg.windows = sorted({labels[c] for c in chosen}, key=lambda w: (w[1] - w[0], w[0]))

    st.subheader("Schätzfenster & Modell")
    e0, e1 = st.slider("Schätzfenster (Handelstage relativ zu t=0)", -500, -1, cfg.estimation_window,
                       help="Standard nach MacKinlay: ca. 250 Tage, mit Abstand zum Event, damit das Ereignis die Schätzung nicht beeinflusst.")
    cfg.estimation_window = (e0, e1)
    st.caption(f"{e1 - e0 + 1} Handelstage, Abstand zum Event: {abs(e1) - 1} Tage")
    models = list(RETURN_MODELS)
    cfg.model = st.radio("Renditemodell (Hauptspezifikation)", models, index=models.index(cfg.model),
                         format_func=RETURN_MODELS.get, horizontal=True)
    st.caption({
        "market": "Standard in der Literatur: R_i = α + β·R_m. Berücksichtigt das Marktrisiko jeder Aktie.",
        "market_adjusted": "Ohne Schätzung: AR = R_i − R_m. Robust, aber unterstellt β = 1.",
        "mean_adjusted": "AR = R_i − durchschnittliche Rendite im Schätzfenster. Ignoriert Marktbewegungen – nur als Robustheitscheck sinnvoll.",
        "ff3": "Markt-, Größen- und Value-Faktor (Fama & French 1993). Faktoren aus der Kenneth R. French Data Library.",
    }[cfg.model])
    cfg.return_type = st.radio("Renditen", ["log", "simple"], index=0 if cfg.return_type == "log" else 1, horizontal=True,
                               format_func={"log": "logarithmiert", "simple": "diskret"}.get)

# ------------------------------------------------------------------- step 5
elif step == 4:
    st.subheader("Robustheit")
    rob = [m for m in RETURN_MODELS if m != cfg.model]
    cfg.robustness_models = st.multiselect("Alternative Renditemodelle", rob, default=[m for m in cfg.robustness_models if m in rob],
                                           format_func=RETURN_MODELS.get,
                                           help="Bleibt der Effekt unter anderen Modellen bestehen, ist er nicht modellgetrieben.")
    c1, c2 = st.columns(2)
    methods = {"none": "Keine", "winsorize": "Winsorisieren", "trim": "Trimmen (entfernen)"}
    cfg.outliers.method = c1.selectbox("Ausreißerbehandlung (Hauptspezifikation)", list(methods),
                                       index=list(methods).index(cfg.outliers.method), format_func=methods.get)
    cfg.outliers.level = c2.slider("Anteil je Rand", 0.005, 0.10, cfg.outliers.level, 0.005, format="%.3f",
                                   disabled=cfg.outliers.method == "none")
    cfg.outlier_robustness = st.checkbox("Zusätzlich Variante mit/ohne Ausreißerbehandlung ausweisen", cfg.outlier_robustness)

    st.markdown("**Placebo-Tests** – dieselbe Analyse an zufälligen Tagen ohne Ereignis.")
    c1, c2, c3, c4 = st.columns(4)
    cfg.placebo.enabled = c1.toggle("aktiv", cfg.placebo.enabled)
    cfg.placebo.n = c2.number_input("Anzahl", 20, 1000, cfg.placebo.n, disabled=not cfg.placebo.enabled)
    cfg.placebo.seed = c3.number_input("Seed", 0, 10_000, cfg.placebo.seed, disabled=not cfg.placebo.enabled)
    cfg.placebo.buffer_days = c4.number_input("Abstand zu Events (Tage)", 10, 250, cfg.placebo.buffer_days,
                                              disabled=not cfg.placebo.enabled)
    names = [g.name for g in cfg.groups]
    cfg.placebo.groups = st.multiselect("Placebo für Gruppen (leer = alle Untersuchungsgruppen)", names,
                                        default=[g for g in cfg.placebo.groups if g in names], disabled=not cfg.placebo.enabled)

    st.subheader("Signifikanztests")
    cfg.tests = st.multiselect("Tests", list(TESTS), default=cfg.tests, format_func=TESTS.get)
    with st.expander("Welche Tests brauche ich?"):
        st.markdown("""
- **t-Test & Wilcoxon** – Standard, wie in den meisten Abschlussarbeiten. Wilcoxon ist robust gegen Ausreißer.
- **BMP** – nutzt standardisierte CARs, robust gegen ereignisbedingt erhöhte Volatilität.
- **Kolari-Pynnönen** – korrigiert BMP für Korrelation zwischen Unternehmen. **Wichtig, wenn alle Firmen dasselbe Event-Datum haben** (z. B. Regulierung): Sonst wird die Signifikanz überschätzt.
- **Rangtest** – nicht-parametrisch über die Zeit, robust gegen schiefe Renditeverteilungen.
- **Patell, Vorzeichentest** – klassische Ergänzungen.
""")
    st.subheader("Datenbereinigung")
    c1, c2, c3 = st.columns(3)
    cfg.min_coverage = c1.slider("Mindest-Datenabdeckung", 0.5, 1.0, cfg.min_coverage, 0.05)
    cfg.min_estimation_obs = c2.number_input("Mindestbeobachtungen im Schätzfenster", 10, 500, cfg.min_estimation_obs)
    cfg.require_full_event_window = c3.checkbox("Vollständige Kurse im Eventfenster verlangen", cfg.require_full_event_window)
    flt = st.checkbox("Extreme Tagesrenditen (|log r| > 1) als Datenfehler entfernen",
                      cfg.extreme_return_threshold is not None,
                      help="Fängt z. B. fehlerhaft bereinigte Splits ab. Jede Änderung wird protokolliert.")
    cfg.extreme_return_threshold = 1.0 if flt else None

# ------------------------------------------------------------------- step 6
else:
    st.subheader("Datenquelle")
    src = st.radio("Kurse", ["yahoo", "upload"], index=0 if cfg.data_source == "yahoo" else 1, horizontal=True,
                   format_func={"yahoo": "Yahoo Finance (automatisch)", "upload": "Eigene CSV hochladen"}.get)
    cfg.data_source = src
    upload = None
    if src == "upload":
        kind = st.radio("Inhalt der Datei", ["Kurse", "Renditen"], horizontal=True)
        upload = st.file_uploader("CSV: erste Spalte Datum, danach eine Spalte je Ticker (inkl. Marktindex)", type=["csv"])
        st.caption("Indizes müssen als Spalte mit ihrem Ticker vorliegen (z. B. `^STOXX`).")

    st.subheader("Methodik-Check")
    checks = check_config(cfg)
    icons = {"error": "⛔", "warning": "⚠️", "hint": "💡"}
    if not checks:
        st.success("Keine Auffälligkeiten.")
    for lvl, msg in checks:
        (st.error if lvl == "error" else st.warning if lvl == "warning" else st.info)(f"{icons[lvl]} {msg}")

    st.subheader("Zusammenfassung")
    st.markdown(
        f"- **Events:** {', '.join(f'{e.date} ({e.label})' for e in cfg.events) or '–'}\n"
        f"- **Gruppen:** {', '.join(f'{g.name} [{len(g.tickers)}]' for g in cfg.groups) or '–'}\n"
        f"- **Fenster:** {', '.join(StudyConfig.window_label(w) for w in cfg.windows)} · Schätzfenster {list(cfg.estimation_window)}\n"
        f"- **Modell:** {RETURN_MODELS[cfg.model]}" + (f" + {', '.join(RETURN_MODELS[m] for m in cfg.robustness_models)}" if cfg.robustness_models else "") + "\n"
        f"- **Tests:** {', '.join(TESTS[t] for t in cfg.tests)}\n"
        f"- **Placebo:** {f'{cfg.placebo.n} Tage' if cfg.placebo.enabled else 'aus'}")

    errors = [m for lvl, m in checks if lvl == "error"]
    a, b = st.columns(2)
    if a.button("💾 Als Entwurf speichern", use_container_width=True):
        st.session_state["draft_slug"] = storage.save_config(cfg, st.session_state.get("draft_slug"))
        st.success("Gespeichert.")
    if b.button("🚀 Event Study durchführen", type="primary", use_container_width=True,
                disabled=bool(errors) or (src == "upload" and upload is None)):
        slug = storage.save_config(cfg, st.session_state.get("draft_slug"))
        st.session_state["draft_slug"] = slug
        if src == "upload":
            frame = read_wide_csv(upload)
            provider = FrameProvider(returns=frame) if kind == "Renditen" else FrameProvider(prices=frame)
        else:
            provider = YahooProvider()
        bar = st.progress(0.0, "Starte …")
        try:
            result = run_study(cfg, provider=provider, progress_cb=lambda f, t: bar.progress(f, t))
        except Exception as exc:  # show any data/network problem to the user
            st.error(f"Die Studie konnte nicht berechnet werden: {exc}")
            st.stop()
        storage.save_result(slug, result)
        open_results(slug)
        st.switch_page("views/results.py")

nav_buttons()
