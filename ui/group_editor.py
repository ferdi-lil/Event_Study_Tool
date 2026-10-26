"""Group builder used for treatment and control groups: manual, guided or AI-assisted."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from eventstudy import ai
from eventstudy.config import FF3_REGIONS, MARKET_INDICES, Group, StudyConfig
from eventstudy.screener import COUNTRIES, REGIONS, SECTORS, UniverseQuery, YahooScreener, \
    industries_by_sector
from eventstudy.universe import (categorical_columns, filter_universe, list_builtin, load_universe,
                                 parse_ticker_text, suggest_index)
from ui import style

GUIDE_TREATMENT = """
**So findest du eine sinnvolle Untersuchungsgruppe**
1. **Wirkungskanal klären:** Über welchen Kanal betrifft das Ereignis Unternehmen (Kosten, Umsatz, Risiko, Wettbewerb)?
2. **Objektives Kriterium wählen:** Branchenklassifikation (GICS/Capital IQ), Region/Sitz, Umsatzanteil … – nicht „nach Gefühl“, sonst ist die Auswahl angreifbar.
3. **Mindestgröße & Liquidität:** z. B. Marktkapitalisierung > 500 Mio. USD, damit Kurse informativ sind.
4. **Betroffenheit abstufen (optional):** Untergruppen (z. B. hohe/mittlere/niedrige Exposition) zeigen, ob der Effekt mit der Betroffenheit steigt – ein starkes Argument für Kausalität.
5. **Kriterium dokumentieren:** Der Text unter „Auswahlkriterium“ landet im Methodik-Protokoll.
"""

GUIDE_CONTROL = """
**Wozu Kontrollgruppen?** Sie schließen alternative Erklärungen aus. Bewährte Logik (wie in der AI-Act-Studie):
- **Gleiche Region, andere Branche** (z. B. EU-Non-Tech): Reagiert der *ganze* Markt? Dann ist es kein ereignisspezifischer Effekt.
- **Gleiche Branche, andere Region** (z. B. US-Tech): Gab es am selben Tag einen *globalen* Sektorschock?
- **Indirekt Betroffene** (z. B. UK-Tech): Gibt es Spillover-Effekte?
Kontrollgruppen sollten ähnlich groß und ähnlich liquide sein und keine Überschneidung mit der Untersuchungsgruppe haben.
"""


ONLINE = "Online-Suche (Yahoo-Screener)"
UPLOAD = "Eigene Datei hochladen …"


def _screener_form(key: str) -> pd.DataFrame | None:
    """Query listed companies by region, sector/industry and size – only when asked."""
    c1, c2, c3 = st.columns([1.2, 1.4, 1.6])
    regions = c1.multiselect("Region / Land", list(REGIONS) + list(COUNTRIES), default=["EU"], key=f"{key}_sc_reg")
    labels = {v: k for k, v in SECTORS.items()}
    sectors = [labels[x] for x in c2.multiselect("Sektoren", list(labels), key=f"{key}_sc_sec")]
    by_sector = industries_by_sector()
    inds = c3.multiselect("Branchen (optional)", sorted({i for s in (sectors or by_sector) for i in by_sector.get(s, [])}),
                          key=f"{key}_sc_ind")
    c1, c2, c3 = st.columns([1, 1, 1.2])
    mcap = c1.number_input("Mindest-Börsenwert (Mio. USD)", 0, 1_000_000, 500, step=100, key=f"{key}_sc_mcap")
    limit = c2.number_input("Max. Unternehmen", 10, 1000, 250, step=10, key=f"{key}_sc_max")
    if c3.button("Suchen", key=f"{key}_sc_go", use_container_width=True, disabled=not regions):
        q = UniverseQuery(regions, sectors=sectors, industries=inds, min_market_cap_usd=mcap * 1e6, max_results=limit)
        bar = st.progress(0.0, "Suche …")
        try:
            sc = YahooScreener()
            df = sc.search(q, progress=lambda f, t: bar.progress(f, t))
            st.session_state[f"{key}_sc_df"] = df
            st.session_state[f"{key}_note"] = q.describe()
            if sc.dropped:
                st.caption(f"{len(sc.dropped)} Treffer verworfen, weil der Firmensitz nicht in der Region liegt.")
        except Exception as exc:  # network, Yahoo changes
            st.error(f"Screener nicht erreichbar: {exc}")
        bar.empty()
    st.caption("Lädt nur Firmenlisten (keine Kurse). Ergebnisse werden 7 Tage zwischengespeichert. "
               "Klassifikation und Börsenwert stammen von Yahoo Finance.")
    return st.session_state.get(f"{key}_sc_df")


def _universe_picker(key: str) -> pd.DataFrame | None:
    builtin = {p.stem: p for p in list_builtin()}
    choice = st.selectbox("Unternehmensuniversum", [ONLINE] + list(builtin) + [UPLOAD], key=f"{key}_uni",
                          help="Online-Suche fragt börsennotierte Unternehmen nach Region, Branche und Größe ab. "
                               "Alternativ eine eigene Liste, z. B. ein Export aus Capital IQ, Refinitiv oder Bloomberg.")
    if choice == ONLINE:
        return _screener_form(key)
    if choice in builtin:
        return load_universe(builtin[choice])
    up = st.file_uploader("CSV mit Unternehmensliste", type=["csv"], key=f"{key}_uni_file")
    if up is None:
        st.caption("Benötigt mindestens eine Ticker-Spalte (Yahoo-Format). Hilfreich: Name, Land, Branche, Marktkapitalisierung.")
        return None
    raw = pd.read_csv(up)
    cols = ["–"] + list(raw.columns)
    st.caption("Spalten zuordnen:")
    c = st.columns(4)
    mapping = {
        "ticker": c[0].selectbox("Ticker", cols, key=f"{key}_m_t", index=cols.index(raw.columns[0])),
        "name": c[1].selectbox("Name", cols, key=f"{key}_m_n"),
        "industry": c[2].selectbox("Branche", cols, key=f"{key}_m_i"),
        "market_cap": c[3].selectbox("Marktkap.", cols, key=f"{key}_m_c"),
    }
    mapping = {k: v for k, v in mapping.items() if v != "–"}
    try:
        return load_universe(raw, mapping)
    except ValueError as exc:
        st.error(str(exc))
        return None


def _universe_info(uni: pd.DataFrame) -> str:
    lines = [f"{len(uni)} Unternehmen."]
    for c in categorical_columns(uni):
        vals = uni[c].dropna().astype(str).value_counts()
        lines.append(f"- {c}: " + ", ".join(f"{v} ({n})" for v, n in vals.head(60).items()))
    return "\n".join(lines)


META_COLUMNS = ["country", "industry", "market_cap", "currency"]


def _candidates_from(df: pd.DataFrame, subgroup_col: str | None) -> pd.DataFrame:
    out = pd.DataFrame({"ticker": df["ticker"], "name": df.get("name", "")})
    out["subgroup"] = df[subgroup_col].astype(str) if subgroup_col else ""
    for c in META_COLUMNS:
        if c in df.columns:
            out[c] = df[c]
    out.insert(0, "Aufnehmen", True)
    return out.reset_index(drop=True)


def _manual(key: str) -> None:
    st.markdown("Ticker im **Yahoo-Finance-Format** (z. B. `SAP.DE`, `ASML.AS`, `MSFT`). "
                "Optional je Zeile mit Semikolon: `Ticker; Name; Untergruppe`.")
    text = st.text_area("Ticker", height=160, key=f"{key}_text", placeholder="SAP.DE; SAP SE; High\nIFX.DE; Infineon; High\nNEM.DE")
    up = st.file_uploader("… oder CSV hochladen (Spalten ticker, name, subgroup)", type=["csv"], key=f"{key}_csv")
    if st.button("Liste übernehmen", key=f"{key}_take_manual"):
        if up is not None:
            df = pd.read_csv(up)
            df.columns = [c.lower() for c in df.columns]
            cand = pd.DataFrame({"ticker": df["ticker"].astype(str).str.strip(),
                                 "name": df.get("name", ""), "subgroup": df.get("subgroup", "")}).fillna("")
        else:
            tickers, names, subs = parse_ticker_text(text)
            cand = pd.DataFrame({"ticker": tickers, "name": [names.get(t, "") for t in tickers],
                                 "subgroup": [subs.get(t, "") for t in tickers]})
        cand.insert(0, "Aufnehmen", True)
        st.session_state[f"{key}_cand"] = cand
        st.session_state[f"{key}_note"] = st.session_state.get(f"{key}_note") or "Manuelle Auswahl"


def _guided(key: str, role: str) -> None:
    st.markdown(GUIDE_TREATMENT if role == "treatment" else GUIDE_CONTROL)
    uni = _universe_picker(key)
    if uni is None:
        return
    filters, desc = {}, []
    cats = categorical_columns(uni)
    cols = st.columns(min(len(cats), 3) or 1)
    for i, c in enumerate(cats):
        opts = sorted(uni[c].dropna().astype(str).unique())
        sel = cols[i % len(cols)].multiselect(c, opts, key=f"{key}_f_{c}")
        if sel:
            filters[c] = sel
            desc.append(f"{c} ∈ {{{', '.join(sel)}}}")
    min_cap = top_n = None
    if "market_cap" in uni.columns:
        c1, c2 = st.columns(2)
        min_cap = c1.number_input("Mindest-Marktkapitalisierung (Einheit wie in der Datei)", 0.0, step=100.0,
                                  key=f"{key}_cap") or None
        top_n = c2.number_input("Nur die N größten je Kategorie (0 = alle)", 0, 500, 0, key=f"{key}_topn") or None
        if min_cap:
            desc.append(f"Marktkapitalisierung ≥ {min_cap:,.0f}")
        if top_n:
            desc.append(f"je Kategorie die {top_n} größten Unternehmen")
    hit = filter_universe(uni, filters, min_cap, top_n, list(filters))
    sub_col = st.selectbox("Untergruppen bilden nach (optional)", ["–"] + cats, key=f"{key}_subcol",
                           help="Z. B. KI-Exposition: Ergebnisse werden dann zusätzlich je Untergruppe ausgewiesen.")
    st.caption(f"**{len(hit)} Unternehmen** erfüllen die Kriterien.")
    st.dataframe(hit.head(200), use_container_width=True, height=220, hide_index=True)
    if st.button(f"{len(hit)} Unternehmen übernehmen", key=f"{key}_take_guided", disabled=hit.empty):
        st.session_state[f"{key}_cand"] = _candidates_from(hit, None if sub_col == "–" else sub_col)
        st.session_state[f"{key}_note"] = "Auswahl aus Universum: " + ("; ".join(desc) or "alle Unternehmen")


def _ai(key: str, role: str, context: str) -> None:
    if not ai.available():
        st.info("Die KI-Hilfe braucht einen Anthropic-API-Key. Lege eine Datei `.env` mit "
                "`ANTHROPIC_API_KEY=…` im Projektordner an und starte die App neu.")
        return
    use_uni = st.checkbox("Universum verwenden (empfohlen, reproduzierbar)", value=True, key=f"{key}_ai_uni")
    uni = _universe_picker(f"{key}_ai") if use_uni else None
    desc = st.text_area("Beschreibe die gewünschte Gruppe", key=f"{key}_ai_desc",
                        placeholder="Europäische Unternehmen, deren Geschäftsmodell stark von KI-Regulierung betroffen ist"
                        if role == "treatment" else "Kontrollgruppe gegen allgemeine Marktbewegungen")
    if st.button("Vorschlag erstellen", key=f"{key}_ai_go", disabled=not desc):
        with st.spinner("Claude denkt nach …"):
            try:
                st.session_state[f"{key}_ai_res"] = ai.suggest_group(desc, role, _universe_info(uni) if uni is not None else "", context)
            except ai.AIUnavailable as exc:
                st.error(str(exc))
    res = st.session_state.get(f"{key}_ai_res")
    if not res:
        return
    st.markdown(f"**Auswahlkriterium:** {res['selection_criteria']}")
    st.markdown(res["rationale"])
    for c in res.get("caveats", []):
        st.caption(f"Hinweis: {c}")
    frames = []
    if uni is not None and res.get("filters"):
        flt = {f["column"]: f["values"] for f in res["filters"] if f["column"] in uni.columns}
        frames.append(filter_universe(uni, flt))
    if res.get("tickers"):
        frames.append(pd.DataFrame(res["tickers"])[["ticker", "name"]])
    if frames:
        hit = pd.concat(frames).drop_duplicates("ticker")
        st.caption(f"{len(hit)} vorgeschlagene Unternehmen – bitte prüfen, Ticker aus KI-Vorschlägen können falsch sein.")
        if st.button("Vorschlag übernehmen", key=f"{key}_ai_take"):
            st.session_state[f"{key}_cand"] = _candidates_from(hit, None)
            st.session_state[f"{key}_note"] = f"KI-gestützte Auswahl, geprüft: {res['selection_criteria']}"
            if res.get("market_index"):
                st.session_state[f"{key}_idx_hint"] = res["market_index"]


def group_editor(cfg: StudyConfig, role: str, key: str) -> None:
    """Renders the builder; on confirm appends a Group to cfg.groups."""
    tabs = st.tabs(["Manuell", "Angeleitet (Universum)", "KI-Assistent"])
    context = "; ".join(f"{e.date} {e.label}" for e in cfg.events)
    if role == "control" and cfg.treatment_groups():
        context += " | Untersuchungsgruppe(n): " + "; ".join(
            f"{g.name} ({g.selection_note})" for g in cfg.treatment_groups())
    with tabs[0]:
        _manual(key)
    with tabs[1]:
        _guided(key, role)
    with tabs[2]:
        _ai(key, role, context)

    cand = st.session_state.get(f"{key}_cand")
    if cand is None or cand.empty:
        return
    st.divider()
    style.section("Auswahl prüfen und Gruppe anlegen")
    edited = st.data_editor(cand, key=f"{key}_editor", use_container_width=True, hide_index=True, height=260,
                            column_config={"Aufnehmen": st.column_config.CheckboxColumn(width="small"),
                                           "subgroup": st.column_config.TextColumn("Untergruppe")})
    chosen = edited[edited["Aufnehmen"]]
    c1, c2, c3 = st.columns([2, 2, 1])
    default_name = {"treatment": "Untersuchungsgruppe", "control": "Kontrollgruppe"}[role]
    name = c1.text_input("Gruppenname", value=f"{default_name} {len(cfg.groups) + 1}", key=f"{key}_name")
    idx_labels = {v: k for k, v in MARKET_INDICES.items()}
    suggested = st.session_state.get(f"{key}_idx_hint") or suggest_index(chosen["ticker"].tolist())
    options = list(MARKET_INDICES.values())
    if suggested not in options:
        options.append(suggested)
    market_index = c2.selectbox("Marktindex (Marktmodell)", options, index=options.index(suggested),
                                format_func=lambda v: f"{idx_labels.get(v, v)} ({v})", key=f"{key}_idx",
                                help="Vorschlag nach Herkunft der Ticker. Der Index sollte den Markt abbilden, an dem die Aktien gehandelt werden.")
    ff3 = c3.selectbox("FF3-Region", list(FF3_REGIONS), key=f"{key}_ff3",
                       index=1 if market_index in ("^GSPC", "^IXIC") else 0)
    note = st.text_area("Auswahlkriterium (für das Protokoll)", value=st.session_state.get(f"{key}_note", ""),
                        key=f"{key}_note_edit", height=70)
    if st.button(f"Gruppe „{name}“ mit {len(chosen)} Unternehmen anlegen", type="primary", key=f"{key}_add",
                 disabled=chosen.empty or any(g.name == name for g in cfg.groups)):
        subs = {r.ticker: str(r.subgroup) for r in chosen.itertuples() if str(r.subgroup) not in ("", "nan", "None")}
        cfg.groups.append(Group(
            name=name, tickers=chosen["ticker"].tolist(), role=role, market_index=market_index, ff3_region=ff3,
            names={r.ticker: str(r.name) for r in chosen.itertuples() if str(r.name) not in ("", "nan")},
            subgroups=subs, selection_note=note,
            meta={r["ticker"]: {c: r[c] for c in META_COLUMNS if c in chosen.columns and pd.notna(r[c])}
                  for _, r in chosen.iterrows()} if any(c in chosen.columns for c in META_COLUMNS) else {}))
        for k in [k for k in st.session_state if k.startswith(key)]:
            del st.session_state[k]
        st.rerun()


def group_list(cfg: StudyConfig, role: str) -> None:
    groups = [g for g in cfg.groups if g.role == role]
    if not groups:
        st.info("Noch keine Gruppe angelegt." if role == "treatment" else "Noch keine Kontrollgruppe angelegt (optional, aber dringend empfohlen).")
    for g in groups:
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            subs = f" · Untergruppen: {', '.join(sorted(set(g.subgroups.values())))}" if g.subgroups else ""
            c1.markdown(f"**{g.name}** – {len(g.tickers)} Unternehmen · Index `{g.market_index}`{subs}")
            c1.caption(g.selection_note or "–")
            if c2.button("Entfernen", key=f"rm_{role}_{g.name}"):
                cfg.groups.remove(g)
                st.rerun()
            with c1.expander("Ticker anzeigen"):
                st.write(", ".join(g.tickers))
