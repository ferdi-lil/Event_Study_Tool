"""Significance tests for cumulative abnormal returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .estimation import EventEstimate


def apply_outliers(cars: pd.DataFrame, method: str, level: float) -> tuple[pd.DataFrame, list[str]]:
    """Winsorise or trim CARs (and SCARs) at `level` per tail. Returns data and affected tickers."""
    if method == "none" or len(cars) < 5 or level <= 0:
        return cars, []
    lo, hi = cars["car"].quantile([level, 1 - level])
    affected = cars.index[(cars["car"] < lo) | (cars["car"] > hi)].tolist()
    if method == "trim":
        return cars.drop(index=affected), affected
    out = cars.copy()
    out["car"] = out["car"].clip(lo, hi)
    slo, shi = cars["scar"].quantile([level, 1 - level])
    out["scar"] = out["scar"].clip(slo, shi)
    return out, affected


def mean_residual_correlation(ar_est: pd.DataFrame) -> float:
    """Average pairwise correlation of estimation-window residuals (Kolari & Pynnönen 2010)."""
    if ar_est.shape[1] < 2:
        return 0.0
    c = ar_est.corr(min_periods=30).to_numpy()
    n = c.shape[0]
    off = c[~np.eye(n, dtype=bool)]
    off = off[~np.isnan(off)]
    return float(off.mean()) if off.size else 0.0


def rank_test(est: EventEstimate, window: tuple[int, int], tickers) -> tuple[float, float]:
    """Multi-day Corrado rank test in the Campbell & Wasley (1993) form."""
    tickers = [t for t in tickers if t in est.ar_event.columns]
    if not tickers:
        return np.nan, np.nan
    full = pd.concat([est.ar_est[tickers], est.ar_event[tickers]])
    ranks = full.rank(axis=0)
    m = full.notna().sum(axis=0)
    U = ranks / (m + 1) - 0.5
    ubar = U.mean(axis=1)
    s_u = np.sqrt(np.nanmean(ubar.to_numpy() ** 2))
    days = [d for d in range(window[0], window[1] + 1) if d in est.ar_event.index]
    ubar_evt = ubar.iloc[len(est.ar_est):]
    ubar_evt = ubar_evt[ubar_evt.index.isin(days)]
    L = len(ubar_evt)
    if s_u == 0 or L == 0:
        return np.nan, np.nan
    z = ubar_evt.sum() / (np.sqrt(L) * s_u)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def run_tests(cars: pd.DataFrame, tests: list[str], est: EventEstimate, window: tuple[int, int],
              rbar: float | None = None) -> dict:
    """Return CAAR plus every requested statistic and p-value (two-sided)."""
    n = len(cars)
    res = {"N": n, "CAAR": np.nan, "Median_CAR": np.nan, "Share_positive": np.nan}
    if n == 0:
        return res
    car = cars["car"].to_numpy()
    res.update(CAAR=car.mean(), Median_CAR=float(np.median(car)), Share_positive=float((car > 0).mean()),
               Std_CAR=car.std(ddof=1) if n > 1 else np.nan)
    if n < 2:
        return res
    scar = cars["scar"].to_numpy()

    if "t_test" in tests:
        se = car.std(ddof=1) / np.sqrt(n)
        t = car.mean() / se if se > 0 else np.nan
        res["t_stat"], res["p_t"] = t, 2 * stats.t.sf(abs(t), n - 1)
    if "wilcoxon" in tests:
        try:
            w = stats.wilcoxon(car)
            res["wilcoxon_stat"], res["p_wilcoxon"] = float(w.statistic), float(w.pvalue)
        except ValueError:
            res["p_wilcoxon"] = np.nan
    if "patell" in tests:
        dof = cars["dof"].to_numpy(dtype=float)
        z = scar.sum() / np.sqrt(np.sum(dof / (dof - 2)))
        res["patell_z"], res["p_patell"] = z, 2 * stats.norm.sf(abs(z))
    t_bmp = None
    if "bmp" in tests or "kolari" in tests:
        sd = scar.std(ddof=1)
        t_bmp = scar.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
    if "bmp" in tests:
        res["bmp_t"], res["p_bmp"] = t_bmp, 2 * stats.t.sf(abs(t_bmp), n - 1)
    if "kolari" in tests:
        r = mean_residual_correlation(est.ar_est[cars.index]) if rbar is None else rbar
        adj = np.sqrt((1 - r) / (1 + (n - 1) * r)) if 1 + (n - 1) * r > 0 else 1.0
        t_kp = t_bmp * adj
        res["rbar"], res["kp_t"], res["p_kolari"] = r, t_kp, 2 * stats.t.sf(abs(t_kp), n - 1)
    if "rank" in tests:
        res["rank_z"], res["p_rank"] = rank_test(est, window, cars.index)
    if "sign" in tests:
        p_hat = float(cars["frac_pos_est"].mean())
        w = int((car > 0).sum())
        z = (w - n * p_hat) / np.sqrt(n * p_hat * (1 - p_hat))
        res["sign_z"], res["p_sign"] = z, 2 * stats.norm.sf(abs(z))
    return res


def difference_test(a: pd.Series, b: pd.Series) -> dict:
    """Treatment vs. control: Welch t-test and Mann-Whitney U on CARs."""
    if len(a) < 2 or len(b) < 2:
        return {}
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"Diff": a.mean() - b.mean(), "welch_t": float(t.statistic), "p_welch": float(t.pvalue),
            "p_mannwhitney": float(u.pvalue)}


P_COLUMNS = {
    "p_t": "t-Test", "p_wilcoxon": "Wilcoxon", "p_patell": "Patell", "p_bmp": "BMP",
    "p_kolari": "Kolari-Pynnönen", "p_rank": "Rangtest", "p_sign": "Vorzeichentest",
}


def stars(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
