"""Price download, return calculation and data cleaning."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import Protocol

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


# ------------------------------------------------------------------ providers
class YahooProvider:
    """Adjusted close prices from Yahoo Finance, cached on disk per ticker."""

    name = "Yahoo Finance (Adjusted Close)"

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir / "prices"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, start, end) -> Path:
        safe = ticker.replace("^", "IDX_").replace("/", "_")
        return self.cache_dir / f"{safe}_{start}_{end}.csv"

    def prices(self, tickers: list[str], start, end) -> pd.DataFrame:
        import yfinance as yf

        frames, missing = {}, []
        for t in tickers:
            p = self._cache_path(t, start, end)
            if p.exists():
                frames[t] = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
            else:
                missing.append(t)
        if missing:
            data = yf.download(missing, start=str(start), end=str(end), progress=False,
                               auto_adjust=True, group_by="column", threads=True)
            if not data.empty:
                close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
                if not isinstance(close, pd.DataFrame):
                    close = close.to_frame()
                if close.shape[1] == 1 and len(missing) == 1:
                    close.columns = missing
                for t in missing:
                    if t in close.columns and close[t].notna().any():
                        s = close[t].dropna()
                        s.to_frame(t).to_csv(self._cache_path(t, start, end))
                        frames[t] = s
        if not frames:
            return pd.DataFrame()
        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df.sort_index()


class FrameProvider:
    """Prices or returns supplied by the user (CSV upload, thesis data, tests)."""

    def __init__(self, prices: pd.DataFrame | None = None, returns: pd.DataFrame | None = None,
                 name: str = "Eigene Daten (Upload)") -> None:
        if prices is None and returns is None:
            raise ValueError("prices or returns required")
        self._prices = prices
        self._returns = returns
        self.name = name

    @property
    def has_returns(self) -> bool:
        return self._returns is not None

    def prices(self, tickers, start, end) -> pd.DataFrame:
        cols = [t for t in tickers if t in self._prices.columns]
        return self._prices.loc[str(start):str(end), cols]

    def returns(self, tickers, start, end) -> pd.DataFrame:
        cols = [t for t in tickers if t in self._returns.columns]
        return self._returns.loc[str(start):str(end), cols]


def read_wide_csv(file) -> pd.DataFrame:
    """Read a wide CSV (date column first, one column per ticker)."""
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.sort_index().apply(pd.to_numeric, errors="coerce")


# ------------------------------------------------------------------- factors
FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def load_ff3(region_file: str, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Daily Fama-French 3 factors (decimal, columns Mkt-RF, SMB, HML, RF)."""
    import requests

    cache = cache_dir / "factors" / region_file.replace(".zip", ".csv")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    r = requests.get(FF_BASE + region_file, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1")
    df = parse_ff_csv(raw)
    df.to_csv(cache)
    return df


def parse_ff_csv(raw: str) -> pd.DataFrame:
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and len(parts[0]) == 8 and parts[0].isdigit():
            try:
                rows.append([parts[0]] + [float(x) for x in parts[1:5]])
            except ValueError:
                continue
    if not rows:
        raise ValueError("Keine Faktordaten in der Datei gefunden")
    df = pd.DataFrame(rows, columns=["Date", "Mkt-RF", "SMB", "HML", "RF"])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    return df.set_index("Date") / 100.0


# -------------------------------------------------------------------- returns
def to_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    prices = prices.where(prices > 0)
    if kind == "log":
        r = np.log(prices / prices.shift(1))
    else:
        r = prices.pct_change(fill_method=None)
    return r.iloc[1:]


def clean_returns(returns: pd.DataFrame, group: str, min_coverage: float,
                  protocol: Protocol, extreme_threshold: float | None = 1.0) -> pd.DataFrame:
    """Drop tickers without data / below coverage and flag implausible returns."""
    step = f"Datenbereinigung – {group}"
    keep = []
    for t in returns.columns:
        s = returns[t]
        if s.notna().sum() == 0:
            protocol.exclude(step, t, "keine Kursdaten verfügbar (delisted oder Ticker unbekannt)")
            continue
        first_valid = s.first_valid_index()
        coverage = s.notna().mean()
        if coverage < min_coverage:
            reason = f"Datenabdeckung {coverage:.0%} < {min_coverage:.0%}"
            if first_valid is not None and first_valid > returns.index[0] + pd.Timedelta(days=30):
                reason += f" (erste Notierung {first_valid.date()}, evtl. IPO nach Beginn des Zeitraums)"
            protocol.exclude(step, t, reason)
            continue
        keep.append(t)
    cleaned = returns[keep].copy()
    # Absolute log returns above 1 (~ +172% / -63% in one day) are almost always data errors.
    if extreme_threshold is None:
        protocol.info(step, f"{len(keep)} von {returns.shape[1]} Unternehmen verbleiben nach Bereinigung")
        return cleaned
    extreme = (cleaned.abs() > extreme_threshold)
    for t in cleaned.columns[extreme.any()]:
        days = list(cleaned.index[extreme[t]].date)
        protocol.warn(step, f"{t}: {len(days)} extreme Tagesrendite(n) auf NaN gesetzt ({', '.join(map(str, days[:3]))})")
        cleaned.loc[extreme[t], t] = np.nan
    protocol.info(step, f"{len(keep)} von {returns.shape[1]} Unternehmen verbleiben nach Bereinigung")
    return cleaned
