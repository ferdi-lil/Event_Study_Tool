"""Result tables, Excel export and a Markdown report."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .config import RETURN_MODELS, TESTS, StudyConfig
from .engine import ALL, MAIN, StudyResult
from .significance import P_COLUMNS, stars


def p_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in P_COLUMNS if c in df.columns and df[c].notna().any()]


def results_table(res: StudyResult, event: str | None = None, model: str | None = None,
                  variant: str = MAIN, subgroups: bool = False, star_col: str = "p_t") -> pd.DataFrame:
    """Thesis-style table: one row per group × window, CAAR in % with significance stars."""
    s = res.summary
    s = s[(s["Model"] == (model or res.config.model)) & (s["Variant"] == variant)]
    if event:
        s = s[s["Event"] == event]
    if not subgroups:
        s = s[s["Subgroup"] == ALL]
    rows = []
    for _, r in s.iterrows():
        row = {"Event": r["Event"], "Gruppe": r["Group"] if r["Subgroup"] == ALL else f"{r['Group']} · {r['Subgroup']}",
               "Fenster": r["Window"], "N": int(r["N"]),
               "CAAR": f"{r['CAAR'] * 100:.2f}%{stars(r.get(star_col, np.nan))}" if pd.notna(r["CAAR"]) else "–"}
        if "t_stat" in r and pd.notna(r.get("t_stat")):
            row["t"] = round(r["t_stat"], 3)
        for c in p_columns(s):
            row[P_COLUMNS[c]] = round(r[c], 4) if pd.notna(r[c]) else None
        row["Anteil CAR>0"] = f"{r['Share_positive']:.0%}" if pd.notna(r.get("Share_positive")) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def model_comparison(res: StudyResult, event: str, star_col: str = "p_t") -> pd.DataFrame:
    s = res.summary
    s = s[(s["Event"] == event) & (s["Subgroup"] == ALL)]
    s = s.assign(Spezifikation=np.where(s["Variant"] == MAIN, s["Model"].map(RETURN_MODELS),
                                        s["Model"].map(RETURN_MODELS) + " · " + s["Variant"]))
    s = s.assign(Wert=[f"{c * 100:.2f}%{stars(p)}" if pd.notna(c) else "–"
                       for c, p in zip(s["CAAR"], s.get(star_col, pd.Series(np.nan, index=s.index)))])
    order = list(dict.fromkeys(s["Window"]))
    t = s.pivot_table(index=["Group", "Spezifikation"], columns="Window", values="Wert", aggfunc="first")
    return t[[w for w in order if w in t.columns]].reset_index()


def to_excel(res: StudyResult) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([{"Feld": k, "Wert": v} for k, v in design_rows(res.config)]).to_excel(xw, sheet_name="Design", index=False)
        res.group_sizes.to_excel(xw, sheet_name="Stichprobe", index=False)
        results_table(res).to_excel(xw, sheet_name="Ergebnisse", index=False)
        res.summary.to_excel(xw, sheet_name="Alle Statistiken", index=False)
        if not res.differences.empty:
            res.differences.to_excel(xw, sheet_name="Gruppenvergleich", index=False)
        if not res.placebo.empty:
            res.placebo.to_excel(xw, sheet_name="Placebo", index=False)
        res.firm_cars.to_excel(xw, sheet_name="CARs je Unternehmen", index=False)
        res.aar.to_excel(xw, sheet_name="AAR-Verlauf", index=False)
        pd.DataFrame(res.protocol.to_records()).to_excel(xw, sheet_name="Protokoll", index=False)
    return buf.getvalue()


def design_rows(cfg: StudyConfig) -> list[tuple[str, str]]:
    rows = [("Studie", cfg.name), ("Beschreibung", cfg.description)]
    for e in cfg.events:
        rows.append(("Event", f"{e.date} – {e.label}"))
    for g in cfg.groups:
        role = "Untersuchungsgruppe" if g.role == "treatment" else "Kontrollgruppe"
        rows.append((role, f"{g.name}: {len(g.tickers)} Unternehmen, Index {g.market_index}. {g.selection_note}"))
    rows += [
        ("Hauptmodell", RETURN_MODELS[cfg.model]),
        ("Robustheitsmodelle", ", ".join(RETURN_MODELS[m] for m in cfg.robustness_models) or "–"),
        ("Schätzfenster", f"[{cfg.estimation_window[0]}, {cfg.estimation_window[1]}] Handelstage"),
        ("Eventfenster", ", ".join(StudyConfig.window_label(w) for w in cfg.windows)),
        ("Renditen", "logarithmiert" if cfg.return_type == "log" else "diskret"),
        ("Tests", ", ".join(TESTS[t] for t in cfg.tests)),
        ("Mindest-Datenabdeckung", f"{cfg.min_coverage:.0%}"),
        ("Vollständiges Eventfenster", "ja" if cfg.require_full_event_window else "nein"),
        ("Ausreißer", f"{cfg.outliers.method} ({cfg.outliers.level:.0%} je Rand)" if cfg.outliers.method != "none" else "keine Behandlung"),
        ("Placebo", f"{cfg.placebo.n} Tage, Seed {cfg.placebo.seed}" if cfg.placebo.enabled else "aus"),
    ]
    return rows


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_keine Daten_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def to_markdown(res: StudyResult, interpretation: str = "") -> str:
    cfg = res.config
    parts = [f"# {cfg.name}", "", cfg.description, "", "## Studiendesign", "",
             _md_table(pd.DataFrame(design_rows(cfg), columns=["Feld", "Wert"])), "",
             "## Stichprobe", "", _md_table(res.group_sizes), ""]
    for e in cfg.events:
        parts += [f"## Ergebnisse {e.date} – {e.label}", "",
                  _md_table(results_table(res, event=e.date).drop(columns=["Event"])), "",
                  "Sterne: * p<0,05; ** p<0,01; *** p<0,001 (t-Test).", ""]
        if any(g.subgroups for g in cfg.groups):
            sub = results_table(res, event=e.date, subgroups=True)
            sub = sub[sub["Gruppe"].str.contains("·")]
            if not sub.empty:
                parts += ["### Untergruppen", "", _md_table(sub.drop(columns=["Event"])), ""]
        if not res.differences.empty:
            d = res.differences[res.differences["Event"] == e.date].copy()
            if not d.empty:
                d["Diff"] = (d["Diff"] * 100).round(2).astype(str) + "%"
                parts += ["### Untersuchungs- vs. Kontrollgruppe", "",
                          _md_table(d.drop(columns=["Event"]).round(4)), ""]
        if cfg.robustness_models or cfg.outlier_robustness:
            parts += ["### Robustheit", "", _md_table(model_comparison(res, e.date)), ""]
        if not res.placebo.empty:
            p = res.placebo[res.placebo["Event"] == e.date].copy()
            if not p.empty:
                for c in ("Actual_CAAR", "Placebo_Mean", "Placebo_Std"):
                    p[c] = (p[c] * 100).round(2).astype(str) + "%"
                parts += ["### Placebo-Tests", "", _md_table(p.drop(columns=["Event"]).round(3)), ""]
    if interpretation:
        parts += ["## Interpretation (KI-Entwurf, bitte prüfen)", "", interpretation, ""]
    parts += ["## Methodik-Protokoll", "", res.protocol.to_markdown(), "",
              f"_Erstellt {res.finished} · Datenquelle: {res.data_source} · Laufzeit {res.runtime_s:.0f} s_"]
    return "\n".join(parts)
