"""Normal-return models and abnormal-return calculation for one group around one event."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import StudyConfig
from .protocol import Protocol


@dataclass
class EventEstimate:
    """Everything one (group, event, model) run produces before testing."""

    event_day: pd.Timestamp
    estimation_range: tuple[pd.Timestamp, pd.Timestamp]
    firms: pd.DataFrame  # one row per firm: params, sigma, n_est, df, frac_pos
    ar_event: pd.DataFrame  # rel. day x ticker
    ar_est: pd.DataFrame  # rel. day x ticker (estimation residuals)
    cars: dict[tuple[int, int], pd.DataFrame] = field(default_factory=dict)  # window -> ticker, car, scar
    excluded: dict[tuple[int, int], list[str]] = field(default_factory=dict)

    def aar(self) -> pd.DataFrame:
        aar = self.ar_event.mean(axis=1)
        return pd.DataFrame({"AAR": aar, "CAAR": aar.cumsum(), "N": self.ar_event.notna().sum(axis=1)})


def trading_calendar(returns: pd.DataFrame, market: pd.Series | None,
                     factors: pd.DataFrame | None = None) -> pd.DatetimeIndex:
    days = returns.dropna(how="all").index
    if market is not None:
        days = days.intersection(market.dropna().index)
    if factors is not None:
        days = days.intersection(factors.dropna().index)
    return days.sort_values()


def locate_event(calendar: pd.DatetimeIndex, event_date: pd.Timestamp) -> int:
    """Index of the first trading day on or after the event date (t=0)."""
    idx = calendar.searchsorted(event_date)
    if idx >= len(calendar):
        raise ValueError(f"Keine Handelstage nach dem Event {event_date.date()} in den Daten")
    return int(idx)


def _design(model: str, dates, market: pd.Series | None, factors: pd.DataFrame | None) -> np.ndarray | None:
    if model == "market":
        return np.column_stack([np.ones(len(dates)), market.reindex(dates).to_numpy()])
    if model == "ff3":
        f = factors.reindex(dates)
        return np.column_stack([np.ones(len(dates)), f[["Mkt-RF", "SMB", "HML"]].to_numpy()])
    return None


def estimate_event(returns: pd.DataFrame, market: pd.Series | None, factors: pd.DataFrame | None,
                   event_date: pd.Timestamp, windows: list[tuple[int, int]],
                   estimation_window: tuple[int, int], model: str, min_obs: int,
                   require_full_window: bool, protocol: Protocol | None = None,
                   step: str = "", calendar: pd.DatetimeIndex | None = None) -> EventEstimate:
    if calendar is None:
        calendar = trading_calendar(returns, market, factors if model == "ff3" else None)
    e = locate_event(calendar, event_date)
    est0, est1 = estimation_window
    lo = min(w[0] for w in windows)
    hi = max(w[1] for w in windows)
    if e + est0 < 0:
        raise ValueError(f"Zu wenig Historie vor {event_date.date()} für ein Schätzfenster von {abs(est0)} Tagen")
    if e + hi >= len(calendar):
        raise ValueError(f"Zu wenig Daten nach {event_date.date()} für das Eventfenster bis +{hi}")

    est_days = calendar[e + est0: e + est1 + 1]
    evt_days = calendar[e + lo: e + hi + 1]
    est_rel = np.arange(est0, est1 + 1)
    evt_rel = np.arange(lo, hi + 1)

    rf_est = rf_evt = None
    if model == "ff3":
        rf_est = factors["RF"].reindex(est_days).to_numpy()
        rf_evt = factors["RF"].reindex(evt_days).to_numpy()
    X_est_full = _design(model, est_days, market, factors)
    X_evt = _design(model, evt_days, market, factors)
    m_est = market.reindex(est_days).to_numpy() if market is not None else None
    m_evt = market.reindex(evt_days).to_numpy() if market is not None else None

    rows, ar_evt_cols, ar_est_cols, var_info = [], {}, {}, {}
    for t in returns.columns:
        y_est = returns[t].reindex(est_days).to_numpy()
        y_evt = returns[t].reindex(evt_days).to_numpy()
        if model == "ff3":
            y_est = y_est - rf_est
            y_evt = y_evt - rf_evt
        ok = ~np.isnan(y_est)
        if X_est_full is not None:
            ok &= ~np.isnan(X_est_full).any(axis=1)
        elif model == "market_adjusted":
            ok &= ~np.isnan(m_est)
        n = int(ok.sum())
        if n < min_obs:
            if protocol:
                protocol.exclude(step, t, f"nur {n} Beobachtungen im Schätzfenster (Minimum {min_obs})")
            continue

        params = {}
        resid_full = np.full(len(est_days), np.nan)
        if model in ("market", "ff3"):
            X = X_est_full[ok]
            beta, *_ = np.linalg.lstsq(X, y_est[ok], rcond=None)
            resid = y_est[ok] - X @ beta
            k = X.shape[1]
            dof = n - k
            sigma = np.sqrt(resid @ resid / dof)
            ar = y_evt - X_evt @ beta
            xtx_inv = np.linalg.pinv(X.T @ X)
            var_info[t] = (sigma, X_evt, xtx_inv)
            names = ["alpha", "beta"] if model == "market" else ["alpha", "beta_mkt", "beta_smb", "beta_hml"]
            params = dict(zip(names, beta))
        elif model == "market_adjusted":
            resid = (y_est - m_est)[ok]
            dof = n - 1
            sigma = resid.std(ddof=1)
            ar = y_evt - m_evt
            var_info[t] = (sigma, None, None)
        else:  # mean_adjusted
            mu = y_est[ok].mean()
            resid = y_est[ok] - mu
            dof = n - 1
            sigma = resid.std(ddof=1)
            ar = y_evt - mu
            var_info[t] = (sigma, None, 1.0 / n)
            params = {"mean": mu}
        resid_full[ok] = resid
        ar_evt_cols[t] = ar
        ar_est_cols[t] = resid_full
        rows.append({"ticker": t, **params, "sigma": sigma, "n_est": n, "dof": dof,
                     "frac_pos_est": float((resid > 0).mean())})

    firms = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
    ar_event = pd.DataFrame(ar_evt_cols, index=evt_rel)
    ar_est = pd.DataFrame(ar_est_cols, index=est_rel)
    result = EventEstimate(calendar[e], (est_days[0], est_days[-1]), firms, ar_event, ar_est)

    for w in windows:
        sel = (evt_rel >= w[0]) & (evt_rel <= w[1])
        L = int(sel.sum())
        recs, excluded = [], []
        for t in ar_event.columns:
            ar_w = ar_event[t].to_numpy()[sel]
            n_missing = int(np.isnan(ar_w).sum())
            if n_missing and (require_full_window or n_missing == L):
                excluded.append(t)
                continue
            car = np.nansum(ar_w)
            sigma, X_evt_t, extra = var_info[t]
            if X_evt_t is not None:
                Xw = X_evt_t[sel]
                v = Xw.sum(axis=0)
                var_car = sigma**2 * (L + v @ extra @ v)
            elif extra is not None:  # mean-adjusted: estimation error of the mean
                var_car = sigma**2 * (L + L**2 * extra)
            else:
                var_car = sigma**2 * L
            recs.append({"ticker": t, "car": car, "scar": car / np.sqrt(var_car),
                         "dof": firms.at[t, "dof"], "frac_pos_est": firms.at[t, "frac_pos_est"],
                         "missing_days": n_missing})
        if excluded and protocol:
            label = StudyConfig.window_label(w)
            if require_full_window:
                protocol.warn(step, f"Fenster {label}: {len(excluded)} Unternehmen ohne vollständige Kurse im Eventfenster ausgeschlossen: {', '.join(excluded)}")
            else:
                protocol.warn(step, f"Fenster {label}: {len(excluded)} Unternehmen ohne jegliche Kurse im Eventfenster ausgeschlossen")
        result.cars[w] = pd.DataFrame(recs).set_index("ticker") if recs else pd.DataFrame(
            columns=["car", "scar", "dof", "frac_pos_est", "missing_days"])
        result.excluded[w] = excluded
    return result
