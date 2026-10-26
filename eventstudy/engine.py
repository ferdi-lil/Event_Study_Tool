"""Runs a complete study: data → cleaning → estimation → tests → robustness → placebo."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from . import data as data_mod
from .checks import check_config
from .config import FF3_REGIONS, RETURN_MODELS, StudyConfig
from .estimation import estimate_event
from .placebo import run_placebo, summarize_placebo
from .protocol import Protocol
from .significance import apply_outliers, difference_test, run_tests

MAIN = "Hauptspezifikation"
ALL = "Alle"


@dataclass
class StudyResult:
    config: StudyConfig
    protocol: Protocol
    summary: pd.DataFrame
    firm_cars: pd.DataFrame
    aar: pd.DataFrame
    differences: pd.DataFrame
    placebo: pd.DataFrame
    placebo_draws: pd.DataFrame
    group_sizes: pd.DataFrame
    checks: list[tuple[str, str]]
    data_source: str
    finished: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    runtime_s: float = 0.0

    def main_table(self) -> pd.DataFrame:
        s = self.summary
        return s[(s["Model"] == self.config.model) & (s["Variant"] == MAIN) & (s["Subgroup"] == ALL)]


class Progress:
    def __init__(self, callback=None):
        self.callback = callback

    def __call__(self, frac: float, text: str) -> None:
        if self.callback:
            self.callback(min(max(frac, 0.0), 1.0), text)


def _log_sources(provider, protocol: Protocol) -> None:
    sources = getattr(provider, "sources", {}) or {}
    if sources:
        counts = pd.Series(list(sources.values())).value_counts()
        protocol.info("Datenbeschaffung", "Herkunft der Kurse: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
        stooq = sorted(t for t, src in sources.items() if "Stooq" in str(src))
        if stooq:
            protocol.warn("Datenbeschaffung", f"{len(stooq)} Ticker nur über Stooq verfügbar – Kurse NICHT "
                                              f"dividendenbereinigt: {', '.join(stooq)}")


def _load_returns(cfg: StudyConfig, provider, protocol: Protocol, progress=None):
    start, end = cfg.data_period()
    indices = sorted({g.market_index for g in cfg.groups})
    all_tickers = sorted({t for g in cfg.groups for t in g.tickers})
    protocol.info("Datenbeschaffung", f"Quelle: {provider.name}; Zeitraum {start} bis {end}")
    protocol.info("Datenbeschaffung", f"{len(all_tickers)} Unternehmen, Marktindizes: {', '.join(indices)}")

    if getattr(provider, "has_returns", False):
        rets = provider.returns(all_tickers + indices, start, end)
        protocol.info("Renditen", "Renditen wurden direkt übernommen (keine Kursumrechnung)")
    else:
        if isinstance(provider, data_mod.YahooProvider) and progress:
            prices = provider.prices(all_tickers + indices, start, end,
                                     progress=lambda f, text: progress(0.02 + 0.03 * f, text))
        else:
            prices = provider.prices(all_tickers + indices, start, end)
        if prices.empty:
            errs = getattr(provider, "errors", {})
            hint = f" Beispiel-Fehler: {next(iter(errs.values()))}" if errs else ""
            raise RuntimeError("Es konnten keine Kursdaten geladen werden. Unter „System“ → „Check starten“ "
                               "lässt sich prüfen, welche Datenquelle nicht erreichbar ist." + hint)
        _log_sources(provider, protocol)
        rets = data_mod.to_returns(prices, cfg.return_type)
        kind = "logarithmierte" if cfg.return_type == "log" else "diskrete"
        protocol.info("Renditen", f"Tägliche {kind} Renditen aus bereinigten Schlusskursen (Dividenden/Splits berücksichtigt)")

    market = {}
    for idx in indices:
        if idx in rets.columns and rets[idx].notna().sum() > 0:
            market[idx] = rets[idx]
        else:
            protocol.warn("Datenbeschaffung", f"Marktindex {idx} konnte nicht geladen werden")
    group_returns, sizes = {}, []
    for g in cfg.groups:
        missing = [t for t in g.tickers if t not in rets.columns]
        for t in missing:
            protocol.exclude(f"Datenbereinigung – {g.name}", t, "keine Kursdaten von der Datenquelle erhalten")
        raw = rets[[t for t in g.tickers if t in rets.columns]]
        cleaned = data_mod.clean_returns(raw, g.name, cfg.min_coverage, protocol,
                                                                 cfg.extreme_return_threshold) if raw.shape[1] else raw
        group_returns[g.name] = cleaned
        sizes.append({"Gruppe": g.name, "Rolle": "Untersuchung" if g.role == "treatment" else "Kontrolle",
                      "Ausgewählt": len(g.tickers), "Mit Daten": raw.shape[1], "Nach Bereinigung": cleaned.shape[1],
                      "Marktindex": g.market_index})
    return group_returns, market, pd.DataFrame(sizes)


def _load_factors(cfg: StudyConfig, protocol: Protocol, factor_loader) -> dict[str, pd.DataFrame]:
    factors = {}
    if "ff3" not in cfg.all_models():
        return factors
    for region in sorted({g.ff3_region for g in cfg.groups}):
        try:
            factors[region] = factor_loader(FF3_REGIONS[region])
            protocol.info("Faktordaten", f"Fama-French-3-Faktoren ({region}, täglich) aus der Kenneth R. French Data Library")
        except Exception as exc:  # network, parse errors
            protocol.warn("Faktordaten", f"FF3-Faktoren für {region} nicht verfügbar ({exc}) – FF3-Modell wird für "
                                         "diese Region übersprungen. Es werden KEINE Ersatzdaten verwendet.")
    return factors


def run_study(cfg: StudyConfig, provider=None, factor_loader=None, progress_cb=None) -> StudyResult:
    t0 = time.time()
    progress = Progress(progress_cb)
    protocol = Protocol()
    checks = check_config(cfg)
    errors = [m for lvl, m in checks if lvl == "error"]
    if errors:
        raise ValueError("Konfiguration unvollständig: " + " ".join(errors))
    for lvl, m in checks:
        if lvl == "warning":
            protocol.warn("Methodik-Check", m)

    provider = provider or data_mod.YahooProvider()
    factor_loader = factor_loader or data_mod.load_ff3
    progress(0.02, "Lade Kursdaten …")
    group_returns, market, sizes = _load_returns(cfg, provider, protocol, progress)
    factors = _load_factors(cfg, protocol, factor_loader)

    protocol.info("Methodik", f"Hauptmodell: {RETURN_MODELS[cfg.model]}; Schätzfenster "
                              f"[{cfg.estimation_window[0]}, {cfg.estimation_window[1]}] Handelstage; "
                              f"Eventfenster: {', '.join(StudyConfig.window_label(w) for w in cfg.windows)}")
    if cfg.robustness_models:
        protocol.info("Methodik", "Robustheitsmodelle: " + ", ".join(RETURN_MODELS[m] for m in cfg.robustness_models))
    protocol.info("Methodik", "Vollständige Kurse im Eventfenster erforderlich" if cfg.require_full_event_window
                  else "Fehlende Tage im Eventfenster werden beim CAR übersprungen")

    variants = [(MAIN, cfg.outliers.method, cfg.outliers.level)]
    if cfg.outlier_robustness:
        if cfg.outliers.method == "none":
            variants.append(("Winsorisiert 1 %", "winsorize", 0.01))
        else:
            variants.append(("Ohne Ausreißerbehandlung", "none", 0.0))

    summary, firm_rows, aar_rows, diff_rows = [], [], [], []
    main_cars: dict = {}
    jobs = [(ev, g, m) for ev in cfg.events for g in cfg.groups for m in cfg.all_models()]
    for j, (ev, g, model) in enumerate(jobs):
        progress(0.05 + 0.55 * j / max(len(jobs), 1), f"Schätze {g.name} · {ev.date} · {model}")
        rets = group_returns.get(g.name)
        mkt = market.get(g.market_index)
        fac = factors.get(g.ff3_region)
        if rets is None or rets.shape[1] == 0:
            continue
        if model in ("market", "market_adjusted") and mkt is None:
            continue
        if model == "ff3" and fac is None:
            continue
        step = f"Schätzung – {g.name} – {ev.date} – {model}"
        try:
            est = estimate_event(rets, mkt, fac, ev.timestamp, cfg.windows, cfg.estimation_window, model,
                                 cfg.min_estimation_obs, cfg.require_full_event_window,
                                 protocol if model == cfg.model else None, step)
        except ValueError as exc:
            protocol.warn(step, str(exc))
            continue
        if model == cfg.model:
            protocol.info(step, f"t=0 ist {est.event_day.date()}; Schätzfenster {est.estimation_range[0].date()} "
                                f"bis {est.estimation_range[1].date()}; {len(est.firms)} Unternehmen geschätzt")
        a = est.aar().reset_index().rename(columns={"index": "rel_day"})
        a.insert(0, "Model", model)
        a.insert(0, "Group", g.name)
        a.insert(0, "Event", ev.date)
        aar_rows.append(a)

        for w, cars in est.cars.items():
            label = StudyConfig.window_label(w)
            for vname, method, level in variants:
                if model != cfg.model and vname != MAIN:
                    continue
                c, affected = apply_outliers(cars, method, level)
                if affected and vname == MAIN and model == cfg.model:
                    verb = "gekappt" if method == "winsorize" else "entfernt"
                    protocol.info(f"Ausreißer – {g.name} – {ev.date}", f"Fenster {label}: {len(affected)} CARs {verb}: {', '.join(affected)}")
                subsets = [(ALL, c)]
                if g.subgroups:
                    for sg in sorted(set(g.subgroups.values())):
                        subsets.append((sg, c[c.index.map(lambda t: g.subgroups.get(t) == sg)]))
                for sg, sub in subsets:
                    res = run_tests(sub, cfg.tests, est, w)
                    summary.append({"Event": ev.date, "Event_Label": ev.label, "Group": g.name, "Role": g.role,
                                    "Subgroup": sg, "Model": model, "Variant": vname, "Window": label,
                                    "w0": w[0], "w1": w[1], **res})
            if model == cfg.model:
                main_cars[(ev.date, g.name, w)] = apply_outliers(cars, cfg.outliers.method, cfg.outliers.level)[0]
                f = cars.copy()
                f["name"] = f.index.map(lambda t: g.names.get(t, ""))
                f["subgroup"] = f.index.map(lambda t: g.subgroups.get(t, ""))
                f = f.reset_index()
                f.insert(0, "Window", label)
                f.insert(0, "Group", g.name)
                f.insert(0, "Event", ev.date)
                firm_rows.append(f)

    # Treatment vs. control differences
    for ev in cfg.events:
        for tg in cfg.treatment_groups():
            for cg in cfg.control_groups():
                for w in cfg.windows:
                    a = main_cars.get((ev.date, tg.name, w))
                    b = main_cars.get((ev.date, cg.name, w))
                    if a is None or b is None:
                        continue
                    d = difference_test(a["car"], b["car"])
                    if d:
                        diff_rows.append({"Event": ev.date, "Treatment": tg.name, "Control": cg.name,
                                          "Window": StudyConfig.window_label(w), **d})

    # Placebo
    placebo_rows, draw_rows = [], []
    if cfg.placebo.enabled:
        targets = cfg.placebo.groups or [g.name for g in cfg.treatment_groups()]
        real = [e.timestamp for e in cfg.events]
        pjobs = [(ev, g) for ev in cfg.events for g in cfg.groups if g.name in targets]
        draws_cache: dict[str, pd.DataFrame] = {}
        for j, (ev, g) in enumerate(pjobs):
            rets = group_returns.get(g.name)
            if rets is None or rets.shape[1] == 0:
                continue
            base = 0.6 + 0.38 * j / max(len(pjobs), 1)
            span = 0.38 / max(len(pjobs), 1)
            progress(base, f"Placebo-Tests {g.name} · {ev.date}")
            # Placebo days are independent of the event, so each group is simulated once.
            draws = draws_cache.get(g.name)
            if draws is None:
                draws = draws_cache[g.name] = run_placebo(rets, market.get(g.market_index), factors.get(g.ff3_region), cfg, cfg.model,
                                    real, cfg.placebo.n, cfg.placebo.seed, cfg.placebo.buffer_days,
                                    progress=lambda f: progress(base + span * f, f"Placebo-Tests {g.name} · {ev.date}"))
            if draws.empty:
                protocol.warn("Placebo", f"{g.name}: keine zulässigen Placebo-Tage gefunden")
                continue
            actual = {}
            for w in cfg.windows:
                c = main_cars.get((ev.date, g.name, w))
                actual[w] = c["car"].mean() if c is not None and len(c) else np.nan
            s = summarize_placebo(draws, actual)
            s["Window"] = s["Window"].map(StudyConfig.window_label)
            s.insert(0, "Group", g.name)
            s.insert(0, "Event", ev.date)
            placebo_rows.append(s)
            d = draws.copy()
            d.insert(0, "Group", g.name)
            d.insert(0, "Event", ev.date)
            draw_rows.append(d)
        if placebo_rows:
            protocol.info("Placebo", f"{cfg.placebo.n} zufällige Placebo-Tage je Gruppe (Seed {cfg.placebo.seed}, "
                                     f"mind. {cfg.placebo.buffer_days} Handelstage Abstand zu echten Events)")

    def cat(rows):
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    draws_df = cat(draw_rows)
    if not draws_df.empty:
        draws_df["Window"] = draws_df["Window"].map(StudyConfig.window_label)
    progress(1.0, "Fertig")
    return StudyResult(cfg, protocol, pd.DataFrame(summary), cat(firm_rows), cat(aar_rows),
                       pd.DataFrame(diff_rows), cat(placebo_rows), draws_df, sizes, checks,
                       provider.name, runtime_s=time.time() - t0)
