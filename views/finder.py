import html
from datetime import date

import pandas as pd
import streamlit as st

from eventstudy import ai
from eventstudy.finder import EVENT_TYPES, PURPOSES, Direction, Idea, ai_ideas, catalog_ideas, idea_to_config
from eventstudy.screener import COUNTRIES, REGIONS, SECTORS, YahooScreener, industries_by_sector
from ui import style
from ui.state import start_new

style.titlebar("Finder", "Ideen für Analysen · Daten werden erst geladen, wenn du eine Idee verfolgst")

# ------------------------------------------------------------------ direction
style.section("Richtung", "Was interessiert dich? Je genauer die Richtung, desto passender die Ideen.")
c1, c2 = st.columns([3, 1.4])
theme = c1.text_input("Thema (frei)", placeholder="z. B. Wie wirkt Klimapolitik auf europäische Windkraft-Hersteller?")
purpose = c2.selectbox("Zweck", PURPOSES)

c1, c2, c3 = st.columns([1.3, 1.7, 1.3])
sector_labels = {v: k for k, v in SECTORS.items()}
sel_sectors = [sector_labels[s] for s in c1.multiselect("Sektoren", list(sector_labels), placeholder="Alle")]
by_sector = industries_by_sector()
ind_options = sorted({i for s in (sel_sectors or by_sector) for i in by_sector.get(s, [])})
sel_inds = c2.multiselect("Branchen (optional, Yahoo-Klassifikation)", ind_options, placeholder="Alle")
regions = c3.multiselect("Region", list(REGIONS) + list(COUNTRIES), default=["EU"], placeholder="Alle")

c1, c2, c3 = st.columns([1, 1, 2.6])
start = c1.date_input("Von", date(2015, 1, 1), format="DD.MM.YYYY")
end = c2.date_input("Bis", date.today(), format="DD.MM.YYYY")
types = c3.multiselect("Ereignisarten", EVENT_TYPES, placeholder="Alle")

direction = Direction(theme=theme, sectors=sel_sectors, industries=sel_inds, regions=regions,
                      start=start.isoformat(), end=end.isoformat(), event_types=types, purpose=purpose)

b1, b2, _ = st.columns([1.3, 1.3, 3])
if b1.button("Ideen aus Katalog", type="primary", use_container_width=True):
    st.session_state["finder_ideas"] = catalog_ideas(direction)
    st.session_state["finder_direction"] = direction
if b2.button("Ideen mit KI", use_container_width=True, disabled=not ai.available(),
             help=None if ai.available() else "Benötigt ANTHROPIC_API_KEY in .env"):
    with st.spinner("Claude sucht passende Ereignisse …"):
        try:
            st.session_state["finder_ideas"] = ai_ideas(direction)
            st.session_state["finder_direction"] = direction
        except ai.AIUnavailable as exc:
            st.error(str(exc))

ideas: list[Idea] = st.session_state.get("finder_ideas", [])
used_direction: Direction = st.session_state.get("finder_direction", direction)
if "finder_checked" not in st.session_state:
    st.session_state["finder_checked"] = set()


def feasibility(n: int) -> tuple[str, str]:
    if n >= 30:
        return "ausreichend", "ok"
    if n >= 10:
        return "knapp", "draft"
    return "zu klein", ""


def resolve_groups(idea: Idea) -> dict[str, pd.DataFrame]:
    """Run the screener for every group of the idea (company lists only, no prices)."""
    key = f"finder_uni_{idea.id}"
    if key in st.session_state:
        return st.session_state[key]
    screener, out, errors = YahooScreener(), {}, []
    bar = st.progress(0.0, "Ermittle Unternehmen …")
    for i, spec in enumerate(idea.groups):
        def prog(f, text, i=i):
            bar.progress((i + f) / len(idea.groups), f"{spec.name}: {text}")
        try:
            out[spec.name] = screener.search(spec.query, progress=prog)
        except Exception as exc:  # network, Yahoo changes
            errors.append(f"{spec.name}: {exc}")
            out[spec.name] = pd.DataFrame()
    bar.empty()
    st.session_state[key] = out
    st.session_state[f"finder_err_{idea.id}"] = errors
    st.session_state["finder_checked"].add(idea.id)
    return out


# --------------------------------------------------------------------- ideas
if ideas:
    style.section(f"Ideen ({len(ideas)})", "Bisher ist nichts berechnet. „Gruppen ermitteln“ lädt nur Firmenlisten; "
                                           "Kursdaten werden erst in der Studie geladen.")
elif "finder_ideas" in st.session_state:
    st.info("Keine passenden Ideen. Zeitraum erweitern, Ereignisarten leeren oder Region/Sektor lockern.")

for idea in ideas:
    with st.container(border=True):
        tags = " ".join([style.tag(idea.origin, "t"), style.tag(idea.event_type),
                         style.tag(f"Überraschung: {idea.surprise}")])
        st.markdown(f'<div class="es-row-name">{html.escape(idea.title)}</div>'
                    f'<div class="es-row-meta">{tags}</div>', unsafe_allow_html=True)
        st.markdown(idea.question)
        left, right = st.columns([1.15, 1])
        with left:
            ev_rows = [f"{pd.Timestamp(e['date']):%d.%m.%Y} · {e.get('label', '')} "
                       f"<span class='es-tag'>{html.escape(e.get('confidence', ''))}</span>" for e in idea.events]
            st.markdown('<div class="es-row-meta"><b>Ereignis</b><br>' + "<br>".join(ev_rows) +
                        f"<br><br><b>Wirkungskanal</b><br>{html.escape(idea.mechanism or '–')}"
                        f"<br><b>Erwartete Richtung:</b> {html.escape(idea.expected_direction)}</div>",
                        unsafe_allow_html=True)
        with right:
            unis = st.session_state.get(f"finder_uni_{idea.id}")
            lines = []
            for g in idea.groups:
                role = "Untersuchung" if g.role == "treatment" else "Kontrolle"
                count = ""
                if unis is not None:
                    n = len(unis.get(g.name, []))
                    label, kind = feasibility(n)
                    count = f" {style.tag(f'{n} Firmen · {label}', kind)}"
                lines.append(f"{style.tag(role, 't' if g.role == 'treatment' else '')} "
                             f"<b>{html.escape(g.name)}</b>{count}<br>"
                             f"<span style='font-size:12px'>{html.escape(g.rationale)}</span>")
            st.markdown('<div class="es-row-meta"><b>Gruppen</b><br>' + "<br>".join(lines) + "</div>",
                        unsafe_allow_html=True)
        warn = []
        if idea.confounders:
            warn.append("Weitere Ereignisse im Umfeld (±7 Tage): " + "; ".join(idea.confounders))
        warn += idea.caveats
        for w in warn:
            st.caption(f"Hinweis: {w}")
        for e in st.session_state.get(f"finder_err_{idea.id}", []):
            st.error(f"Unternehmen konnten nicht ermittelt werden – {e}")

        a, b, _ = st.columns([1.2, 1.2, 3])
        if a.button("Gruppen ermitteln", key=f"res_{idea.id}", use_container_width=True):
            resolve_groups(idea)
            st.rerun()
        if b.button("Als Studie öffnen", key=f"open_{idea.id}", type="primary", use_container_width=True):
            unis = resolve_groups(idea)
            cfg = idea_to_config(idea, unis, used_direction)
            n = len(st.session_state["finder_checked"])
            cfg.description += f" Im Finder wurden in dieser Sitzung {n} Idee(n) näher geprüft."
            start_new(cfg)
            st.switch_page("views/wizard.py")
        if unis is not None:
            with st.expander("Gefundene Unternehmen"):
                for g in idea.groups:
                    df = unis.get(g.name, pd.DataFrame())
                    st.markdown(f"**{g.name}** · {len(df)} Unternehmen")
                    if not df.empty:
                        show = df[["ticker", "name", "country", "industry", "market_cap", "currency"]].copy()
                        show["market_cap"] = (show["market_cap"] / 1e9).round(1)
                        st.dataframe(show.rename(columns={"market_cap": "Börsenwert (Mrd.)"}), hide_index=True,
                                     use_container_width=True, height=min(35 * len(show) + 38, 260))

if not ideas:
    st.markdown('<div class="es-note" style="margin-top:12px">So funktioniert der Finder: 1. Richtung wählen → '
                '2. Ideen ansehen (ohne Daten) → 3. „Gruppen ermitteln“ lädt nur Firmenlisten → '
                '4. „Als Studie öffnen“ übergibt alles an die Werkstatt, dort wird gerechnet.</div>',
                unsafe_allow_html=True)
