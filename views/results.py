import numpy as np
import pandas as pd
import streamlit as st

from eventstudy import ai, storage
from eventstudy.config import RETURN_MODELS, StudyConfig
from eventstudy.engine import ALL, MAIN
from eventstudy.report import model_comparison, results_table, to_excel, to_markdown
from eventstudy.significance import P_COLUMNS, stars
from ui import charts
from ui.state import start_new

studies = [s for s in storage.list_studies() if s["status"] == "Ausgewertet"]
if not studies:
    st.info("Noch keine ausgewerteten Studien.")
    st.stop()
slugs = [s["slug"] for s in studies]
current = st.session_state.get("open_slug")
slug = st.selectbox("Studie", slugs, index=slugs.index(current) if current in slugs else 0,
                    format_func=lambda x: next(s["config"].name for s in studies if s["slug"] == x))
st.session_state["open_slug"] = slug
res = storage.load_result(slug)
cfg: StudyConfig = res.config
colors = charts.group_colors([g.name for g in cfg.groups])
roles = {g.name: g.role for g in cfg.groups}

st.title(cfg.name)
if cfg.description:
    st.caption(cfg.description)
st.caption(f"Berechnet {res.finished.replace('T', ' ')} · {res.data_source} · {res.runtime_s:.0f} s · "
           f"Modell: {RETURN_MODELS[cfg.model]}")
a, b, _ = st.columns([1, 1, 4])
if a.button("✏️ Bearbeiten"):
    start_new(StudyConfig.from_dict(cfg.to_dict()), slug)
    st.switch_page("views/wizard.py")
if b.button("🔁 Als Kopie neu aufsetzen"):
    copy = StudyConfig.from_dict(cfg.to_dict())
    copy.name += " (Variante)"
    start_new(copy)
    st.switch_page("views/wizard.py")

events = [e.date for e in cfg.events]
labels = {e.date: f"{e.date} · {e.label}" if e.label else e.date for e in cfg.events}
c1, c2 = st.columns([2, 2])
event = c1.selectbox("Event", events, format_func=labels.get) if len(events) > 1 else events[0]
p_avail = [c for c in P_COLUMNS if c in res.summary.columns and res.summary[c].notna().any()]
star_col = c2.selectbox("Sterne nach Test", p_avail, format_func=P_COLUMNS.get,
                        help="Signifikanzsterne: * p<0,05, ** p<0,01, *** p<0,001")

main = res.main_table()
main = main[main["Event"] == event]

# ------------------------------------------------------------- key findings
st.subheader("Auf einen Blick")
head_w = StudyConfig.window_label(cfg.windows[0])
tiles = main[main["Window"] == head_w]
cols = st.columns(max(1, min(len(tiles), 5)))
for i, (_, r) in enumerate(tiles.head(5).iterrows()):
    p = r.get(star_col, np.nan)
    sig = "signifikant" if pd.notna(p) and p < 0.05 else "nicht signifikant"
    cols[i].metric(f"{'🎯' if r['Role'] == 'treatment' else '⚖️'} {r['Group']} {head_w}",
                   f"{r['CAAR'] * 100:+.2f} %{stars(p)}", help=f"N={int(r['N'])}, p ({P_COLUMNS.get(star_col)}) = {p:.4f}",
                   delta=f"{sig} (p={p:.3f})", delta_color="off", delta_arrow="off")

kp_gap = pd.DataFrame()
if {"p_t", "p_kolari"} <= set(main.columns):
    kp_gap = main[(main["Role"] == "treatment") & (main["p_t"] < 0.05) & (main["p_kolari"] >= 0.05)]
if not kp_gap.empty:
    st.warning(f"⚠️ In {len(kp_gap)} Fenster(n) ist der t-Test signifikant, der Kolari-Pynnönen-Test (korrigiert für "
               "Korrelation zwischen Unternehmen am selben Event-Tag) aber nicht: "
               + ", ".join(f"{r.Group} {r.Window}" for r in kp_gap.itertuples())
               + ". Diese Ergebnisse vorsichtig interpretieren.")

tabs = st.tabs(["📊 Überblick", "📈 CAAR-Verlauf", "⚖️ Gruppenvergleich", "🏢 Unternehmen", "🧪 Robustheit",
                "🎲 Placebo", "📝 Protokoll", "⬇️ Export & KI"])

with tabs[0]:
    st.plotly_chart(charts.caar_bars(main, colors, f"CAAR je Eventfenster – {labels[event]}"), use_container_width=True)
    st.dataframe(results_table(res, event=event, star_col=star_col).drop(columns=["Event"]),
                 use_container_width=True, hide_index=True)
    sub = results_table(res, event=event, subgroups=True, star_col=star_col)
    sub = sub[sub["Gruppe"].str.contains("·")]
    if not sub.empty:
        st.markdown("**Untergruppen**")
        st.dataframe(sub.drop(columns=["Event"]), use_container_width=True, hide_index=True)
    st.caption("CAAR = durchschnittliche kumulierte abnormale Rendite. Sterne: * p<0,05 · ** p<0,01 · *** p<0,001.")

with tabs[1]:
    aar = res.aar[(res.aar["Event"] == event) & (res.aar["Model"] == cfg.model)]
    sel = st.multiselect("Gruppen", list(aar["Group"].unique()), default=list(aar["Group"].unique()))
    st.plotly_chart(charts.caar_path(aar[aar["Group"].isin(sel)], colors,
                                     f"Kumulierte abnormale Rendite um das Event – {labels[event]}"),
                    use_container_width=True)
    st.caption("Kumuliert ab dem ersten Tag des breitesten Eventfensters. Hover zeigt die tägliche AAR und N.")
    st.dataframe(aar[aar["Group"].isin(sel)].pivot(index="rel_day", columns="Group", values="CAAR").mul(100).round(3),
                 use_container_width=True)

with tabs[2]:
    if res.differences.empty:
        st.info("Kein Vergleich möglich – es gibt keine Kontrollgruppe.")
    else:
        d = res.differences[res.differences["Event"] == event].copy()
        d["Diff"] = d["Diff"].map(lambda x: f"{x * 100:+.2f} %")
        st.markdown("Unterschied der mittleren CARs zwischen Untersuchungs- und Kontrollgruppe "
                    "(Welch-t-Test und Mann-Whitney-U-Test).")
        st.dataframe(d.drop(columns=["Event"]).round(4), use_container_width=True, hide_index=True)
    st.dataframe(res.group_sizes, use_container_width=True, hide_index=True)

with tabs[3]:
    f = res.firm_cars[res.firm_cars["Event"] == event]
    c1, c2 = st.columns(2)
    g = c1.selectbox("Gruppe", list(f["Group"].unique()))
    w = c2.selectbox("Fenster", list(f["Window"].unique()))
    fg = f[(f["Group"] == g) & (f["Window"] == w)].sort_values("car")
    st.plotly_chart(charts.car_histogram(fg["car"], colors.get(g, charts.MUTED), f"Verteilung der CARs – {g} {w}"),
                    use_container_width=True)
    show = fg[["ticker", "name", "subgroup", "car", "scar", "missing_days"]].copy()
    show["car"] = (show["car"] * 100).round(2)
    st.dataframe(show.rename(columns={"car": "CAR %", "scar": "SCAR", "missing_days": "fehlende Tage"}),
                 use_container_width=True, hide_index=True, height=400)

with tabs[4]:
    st.markdown("Bleibt das Ergebnis unter anderen Modellen und mit Ausreißerbehandlung bestehen?")
    st.dataframe(model_comparison(res, event, star_col), use_container_width=True, hide_index=True)
    s = res.summary[(res.summary["Event"] == event) & (res.summary["Subgroup"] == ALL)
                    & (res.summary["Model"] == cfg.model) & (res.summary["Variant"] == MAIN)]
    pcols = [c for c in p_avail if c in s.columns]
    if pcols:
        st.markdown("**Alle p-Werte der Hauptspezifikation**")
        pv = s[["Group", "Window"] + pcols].rename(columns=P_COLUMNS)
        st.dataframe(pv.style.format(precision=4).map(
            lambda v: "font-weight: 700" if isinstance(v, float) and v < 0.05 else "", subset=[P_COLUMNS[c] for c in pcols]),
            use_container_width=True, hide_index=True)
        if "rbar" in s.columns:
            st.caption("Mittlere Residuenkorrelation (Kolari-Pynnönen): "
                       + ", ".join(f"{r.Group}: {r.rbar:.3f}" for r in s.drop_duplicates("Group").itertuples()))

with tabs[5]:
    if res.placebo.empty:
        st.info("Keine Placebo-Tests in dieser Studie.")
    else:
        p = res.placebo[res.placebo["Event"] == event]
        st.markdown("Wie ungewöhnlich ist die tatsächliche CAAR im Vergleich zu zufälligen Tagen ohne Ereignis? "
                    "**Perzentil** nahe 0 oder 100 und **p (empirisch)** < 0,05 sprechen für einen echten Effekt.")
        show = p.drop(columns=["Event"]).copy()
        for c in ("Actual_CAAR", "Placebo_Mean", "Placebo_Std"):
            show[c] = (show[c] * 100).round(2)
        st.dataframe(show.rename(columns={"Group": "Gruppe", "Window": "Fenster", "Actual_CAAR": "CAAR %",
                                          "N_Placebo": "Placebos", "Placebo_Mean": "Placebo-Mittel %",
                                          "Placebo_Std": "Placebo-Std %", "Percentile": "Perzentil",
                                          "p_empirisch": "p (empirisch)", "n_sig_t": "signif. Placebos (t)",
                                          "n_sig_wilcoxon": "signif. Placebos (Wilcoxon)"}),
                     use_container_width=True, hide_index=True)
        st.caption("„signif. Placebos“: An wie vielen Zufallstagen der Test bei 5 % anschlägt. Erwartet sind ~5; "
                   "deutlich mehr heißt, der Test ist für diese Stichprobe zu liberal (meist Korrelation zwischen Firmen).")
        c1, c2 = st.columns(2)
        pg = c1.selectbox("Gruppe", list(p["Group"].unique()), key="pl_g")
        pw = c2.selectbox("Fenster", list(p["Window"].unique()), key="pl_w")
        draws = res.placebo_draws[(res.placebo_draws["Event"] == event) & (res.placebo_draws["Group"] == pg)
                                  & (res.placebo_draws["Window"] == pw)]
        actual = p[(p["Group"] == pg) & (p["Window"] == pw)]["Actual_CAAR"]
        if not draws.empty and not actual.empty:
            st.plotly_chart(charts.placebo_histogram(draws["CAAR"], float(actual.iloc[0]), colors.get(pg, charts.MUTED),
                                                     f"Placebo-Verteilung – {pg} {pw}"), use_container_width=True)

with tabs[6]:
    excl = res.protocol.exclusions()
    warn = res.protocol.warnings()
    c1, c2, c3 = st.columns(3)
    c1.metric("Protokolleinträge", len(res.protocol.entries))
    c2.metric("Ausgeschlossene Unternehmen", len({e.details.get("ticker") for e in excl}))
    c3.metric("Warnungen", len(warn))
    level = st.multiselect("Anzeigen", ["info", "warning", "exclusion"], default=["warning", "exclusion", "info"])
    log = pd.DataFrame(res.protocol.to_records())
    st.dataframe(log[log["Typ"].isin(level)], use_container_width=True, hide_index=True, height=450)

with tabs[7]:
    interp = st.session_state.get(f"interp_{slug}", "")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Excel (alle Tabellen)", to_excel(res), f"{cfg.slug}.xlsx", use_container_width=True)
    c2.download_button("Bericht (Markdown)", to_markdown(res, interp), f"{cfg.slug}.md", use_container_width=True)
    import json
    c3.download_button("Konfiguration (JSON)", json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False),
                       f"{cfg.slug}.json", use_container_width=True)
    st.divider()
    st.markdown("**✨ Ergebnisinterpretation (Entwurf)**")
    if not ai.available():
        st.info("Für einen KI-Entwurf `ANTHROPIC_API_KEY` in `.env` setzen.")
    elif st.button("Interpretation entwerfen"):
        text = to_markdown(res)
        with st.spinner("Claude schreibt …"):
            try:
                st.session_state[f"interp_{slug}"] = ai.interpret_results(f"{cfg.name}. {cfg.description}", text[:60000])
                st.rerun()
            except ai.AIUnavailable as exc:
                st.error(str(exc))
    if interp:
        st.markdown(interp)
        st.caption("KI-Entwurf – inhaltlich prüfen, bevor du ihn verwendest. Er ist im Markdown-Bericht enthalten.")
