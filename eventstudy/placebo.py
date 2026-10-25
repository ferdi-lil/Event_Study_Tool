"""Placebo / randomization inference: rerun the study on random non-event days."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .estimation import estimate_event, locate_event, trading_calendar
from .significance import run_tests


def eligible_days(calendar: pd.DatetimeIndex, real_events: list[pd.Timestamp],
                  estimation_window: tuple[int, int], windows, buffer: int) -> np.ndarray:
    hi = max(w[1] for w in windows)
    idx = np.arange(-estimation_window[0], len(calendar) - hi)
    for ev in real_events:
        try:
            e = locate_event(calendar, ev)
        except ValueError:
            continue
        idx = idx[np.abs(idx - e) > buffer]
    return idx


def run_placebo(returns, market, factors, cfg, model: str, real_events: list[pd.Timestamp],
                n: int, seed: int, buffer: int, progress=None) -> pd.DataFrame:
    calendar = trading_calendar(returns, market, factors if model == "ff3" else None)
    pool = eligible_days(calendar, real_events, cfg.estimation_window, cfg.windows, buffer)
    if len(pool) == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    picks = np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))
    rows = []
    for k, i in enumerate(picks):
        day = calendar[i]
        try:
            est = estimate_event(returns, market, factors, day, cfg.windows, cfg.estimation_window, model,
                                 cfg.min_estimation_obs, cfg.require_full_event_window, calendar=calendar)
        except ValueError:
            continue
        for w, cars in est.cars.items():
            r = run_tests(cars, ["t_test", "wilcoxon"], est, w)
            rows.append({"placebo_date": day.date(), "Window": w, "CAAR": r["CAAR"], "N": r["N"],
                         "p_t": r.get("p_t"), "p_wilcoxon": r.get("p_wilcoxon")})
        if progress:
            progress((k + 1) / len(picks))
    return pd.DataFrame(rows)


def summarize_placebo(draws: pd.DataFrame, actual: dict[tuple[int, int], float]) -> pd.DataFrame:
    out = []
    for w, a in actual.items():
        d = draws[draws["Window"] == w]
        if d.empty or a is None or np.isnan(a):
            continue
        c = d["CAAR"].to_numpy()
        out.append({
            "Window": w, "Actual_CAAR": a, "N_Placebo": len(c), "Placebo_Mean": c.mean(),
            "Placebo_Std": c.std(ddof=1), "Percentile": 100 * (c < a).mean(),
            "p_empirisch": float((np.abs(c) >= abs(a)).mean()),
            "n_sig_t": int((d["p_t"] < 0.05).sum()), "n_sig_wilcoxon": int((d["p_wilcoxon"] < 0.05).sum()),
        })
    return pd.DataFrame(out)
