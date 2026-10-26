"""Price download, return calculation and data cleaning."""
from __future__ import annotations

import contextlib
import io
import json
import logging
import shutil
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import Protocol

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


# ------------------------------------------------------------------ providers
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "application/json,text/csv,*/*",
}


def fetch_yahoo_chart(ticker: str, start, end, session=None) -> pd.Series:
    """Adjusted close straight from Yahoo's chart endpoint – the route R's quantmod uses.

    Works in many setups where the `yfinance` package fails (outdated version, cookie/crumb
    changes, rate limiting), because it needs neither cookies nor a crumb.
    """
    import requests

    sess = session or requests.Session()
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp())
    last_exc = None
    for host in ("query1", "query2"):
        url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(ticker)}"
        try:
            r = sess.get(url, params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,splits",
                                      "includeAdjustedClose": "true"}, headers=BROWSER_HEADERS, timeout=15)
            r.raise_for_status()
            return parse_yahoo_chart(r.json())
        except Exception as exc:  # try the other host
            last_exc = exc
    raise RuntimeError(f"Yahoo-Chart-API: {last_exc}")


def parse_yahoo_chart(payload: dict) -> pd.Series:
    res = (payload.get("chart") or {}).get("result") or []
    if not res or not res[0].get("timestamp"):
        raise ValueError("leere Antwort")
    r = res[0]
    offset = r.get("meta", {}).get("gmtoffset", 0)
    idx = pd.to_datetime(np.array(r["timestamp"]) + offset, unit="s").normalize()
    ind = r["indicators"]
    adj = (ind.get("adjclose") or [{}])[0].get("adjclose") or ind["quote"][0]["close"]
    s = pd.Series(adj, index=idx, dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")]


STOOQ_INDEX = {"^GSPC": "^spx", "^GDAXI": "^dax", "^FCHI": "^cac", "^FTSE": "^ukx", "^N225": "^nkx",
               "^IXIC": "^ndq"}


def stooq_symbol(ticker: str) -> str | None:
    if ticker.startswith("^"):
        return STOOQ_INDEX.get(ticker)
    if "." not in ticker:
        return f"{ticker.lower()}.us"
    base, suffix = ticker.rsplit(".", 1)
    known = {"DE": "de", "F": "de", "L": "uk", "PA": "fr", "AS": "nl", "MI": "it", "MC": "es", "BR": "be",
             "VI": "at", "SW": "ch", "HE": "fi", "ST": "se", "CO": "dk", "OL": "no", "WA": "pl", "T": "jp", "HK": "hk"}
    return f"{base.lower()}.{known[suffix.upper()]}" if suffix.upper() in known else None


def fetch_stooq(ticker: str, start, end, session=None) -> pd.Series:
    """Close prices from stooq.com – last resort, NOT dividend-adjusted."""
    import requests

    sym = stooq_symbol(ticker)
    if sym is None:
        raise ValueError("kein Stooq-Symbol bekannt")
    sess = session or requests.Session()
    r = sess.get("https://stooq.com/q/d/l/", params={"s": sym, "i": "d", "d1": pd.Timestamp(start).strftime("%Y%m%d"),
                                                      "d2": pd.Timestamp(end).strftime("%Y%m%d")},
                 headers=BROWSER_HEADERS, timeout=15)
    r.raise_for_status()
    if not r.text.startswith("Date"):
        raise ValueError(r.text.strip()[:80] or "keine Daten")
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
    return df["Close"].astype(float).dropna()


class _Collect(logging.Handler):
    """Keeps yfinance's error messages so they can be shown in the app instead of the console."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record) -> None:
        self.messages.append(record.getMessage().strip())


class YahooProvider:
    """Adjusted close prices, cached on disk per ticker, with a chain of fallbacks:

    1. `yfinance` package, 2. Yahoo chart API directly (like R/quantmod), 3. Stooq (unadjusted).
    `self.sources` records where every ticker came from; it ends up in the protocol.
    """

    name = "Yahoo Finance (Adjusted Close)"

    def __init__(self, cache_dir: Path = CACHE_DIR, use_stooq: bool = True) -> None:
        self.cache_dir = cache_dir / "prices"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_stooq = use_stooq
        self.sources: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self.systemic_failure = False

    def _cache_path(self, ticker: str, start, end) -> Path:
        safe = ticker.replace("^", "IDX_").replace("/", "_")
        return self.cache_dir / f"{safe}_{start}_{end}.csv"

    def _store(self, frames, t, s, start, end, source):
        s = s.dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        out = s.to_frame(t)
        out["source"] = source
        out.to_csv(self._cache_path(t, start, end))
        frames[t] = s
        self.sources[t] = source

    def _yfinance(self, tickers, start, end, report=None, batch: int = 40) -> dict[str, pd.Series]:
        """Download in batches (large single requests trigger Yahoo's rate limit), one retry per batch."""
        try:
            import yfinance as yf
        except ImportError as exc:
            self.errors.update({t: f"yfinance nicht installiert: {exc}" for t in tickers})
            return {}
        # yfinance floods the console with pandas deprecation warnings and per-ticker error
        # lines; we collect its error log instead and report missing tickers ourselves.
        logger = logging.getLogger("yfinance")
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        collected = _Collect()
        logger.addHandler(collected)
        try:
            return self._yfinance_batches(yf, tickers, start, end, report, batch, collected)
        finally:
            logger.removeHandler(collected)

    def _yfinance_batches(self, yf, tickers, start, end, report, batch, collected) -> dict[str, pd.Series]:
        out = {}
        chunks = [tickers[i:i + batch] for i in range(0, len(tickers), batch)]
        for k, chunk in enumerate(chunks):
            if report:
                report(k / len(chunks), f"yfinance: Ticker {k * batch + 1}–{k * batch + len(chunk)} von {len(tickers)} …")
            for attempt in range(2):
                collected.messages.clear()
                captured = io.StringIO()
                try:
                    with warnings.catch_warnings(), contextlib.redirect_stdout(captured), \
                            contextlib.redirect_stderr(captured):
                        warnings.simplefilter("ignore")
                        data = yf.download(chunk, start=str(start), end=str(end), progress=False,
                                           auto_adjust=True, group_by="column", threads=True)
                except Exception as exc:  # rate limit, network
                    self.errors.update({t: f"yfinance: {exc}" for t in chunk})
                    data = None
                if data is not None and not data.empty:
                    break
                # yfinance prints its reason instead of raising – keep the gist for the user
                lines = collected.messages + [ln.strip() for ln in captured.getvalue().splitlines()
                                              if ln.strip() and "Warning" not in ln and "inplace" not in ln
                                              and "pandas" not in ln]
                if lines:
                    self.errors.update({t: f"yfinance: {lines[-1][:200]}" for t in chunk})
                time.sleep(2)
            if data is None or data.empty:
                continue
            close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
            if not isinstance(close, pd.DataFrame):
                close = close.to_frame()
            if close.shape[1] == 1 and len(chunk) == 1:
                close.columns = chunk
            out.update({t: close[t] for t in chunk if t in close.columns and close[t].notna().any()})
        return out

    def _failed_path(self, start, end) -> Path:
        return self.cache_dir / f"_nicht_verfuegbar_{start}_{end}.json"

    def _known_failures(self, start, end) -> dict[str, str]:
        """Tickers that found no source within the last 24 h – skipped instead of retried."""
        p = self._failed_path(start, end)
        if not p.exists() or (pd.Timestamp.now() - pd.Timestamp(p.stat().st_mtime, unit="s")) > pd.Timedelta(hours=24):
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return {}

    def prices(self, tickers: list[str], start, end, progress=None) -> pd.DataFrame:
        """`progress(fraction, text)` is called while downloading."""
        import requests

        def report(frac, text):
            if progress:
                progress(frac, text)

        frames, missing = {}, []
        failed_before = self._known_failures(start, end)
        for t in tickers:
            p = self._cache_path(t, start, end)
            if p.exists():
                c = pd.read_csv(p, index_col=0, parse_dates=True)
                frames[t] = c.iloc[:, 0]
                self.sources[t] = (c["source"].iloc[0] if "source" in c.columns and len(c) else "Cache") + " (Cache)"
            elif t in failed_before:
                self.errors[t] = f"bereits zuvor ohne Daten (wird 24 h nicht erneut versucht): {failed_before[t]}"
            else:
                missing.append(t)
        report(0.0, f"{len(frames)} Ticker aus dem Cache, {len(missing)} werden online geladen …")
        if missing:
            got = self._yfinance(missing, start, end, report=lambda f, text: report(0.6 * f, text))
            for t, s in got.items():
                self._store(frames, t, s, start, end, "yfinance")
        sess = requests.Session()
        rest = [t for t in missing if t not in frames]
        for i, t in enumerate(rest):
            report(0.6 + 0.4 * i / max(len(rest), 1), f"Ersatzquellen für {t} ({i + 1}/{len(rest)}) …")
            try:
                self._store(frames, t, fetch_yahoo_chart(t, start, end, sess), start, end, "Yahoo-Chart-API")
                continue
            except Exception as exc:
                self.errors[t] = str(exc)
            if self.use_stooq:
                try:
                    self._store(frames, t, fetch_stooq(t, start, end, sess), start, end, "Stooq (unbereinigt)")
                except Exception as exc:
                    self.errors[t] += f" | Stooq: {exc}"
        failed = {t: self.errors.get(t, "")[:200] for t in rest if t not in frames}
        fetched_online = len([t for t in missing if t in frames])
        # Remember failures only if the sources basically worked; if nothing came through at all
        # (network down, rate limit), blacklisting every ticker would block the next attempt.
        self.systemic_failure = bool(missing) and fetched_online == 0 and len(missing) > 3
        if failed and not self.systemic_failure:
            self._failed_path(start, end).write_text(json.dumps({**failed_before, **failed}, ensure_ascii=False),
                                                     encoding="utf-8")
        report(1.0, "Kursdaten geladen")
        if not frames:
            return pd.DataFrame()
        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        return df.sort_index()


def clear_price_cache(only_failures: bool = False, cache_dir: Path = CACHE_DIR) -> int:
    """Delete cached prices (or just the 24-h 'not available' lists). Returns number of files removed."""
    d = cache_dir / "prices"
    if not d.exists():
        return 0
    files = list(d.glob("_nicht_verfuegbar_*.json")) if only_failures else list(d.iterdir())
    for f in files:
        f.unlink() if f.is_file() else shutil.rmtree(f)
    return len(files)


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


class BundleProvider(FrameProvider):
    """Prices saved by scripts/fetch_data.py (or the R script) in data/<name>/prices.csv."""

    def __init__(self, folder: Path) -> None:
        folder = Path(folder)
        super().__init__(prices=read_wide_csv(folder / "prices.csv"), name=f"Datenpaket {folder.name}")
        self.folder = folder
        src = folder / "sources.csv"
        self.sources = pd.read_csv(src, index_col=0).iloc[:, 0].to_dict() if src.exists() else {}

    def factor_loader(self, region_file: str) -> pd.DataFrame:
        stem = region_file.replace("_CSV.zip", "")
        p = self.folder / f"ff3_{stem}.csv"
        if p.exists():
            return pd.read_csv(p, index_col=0, parse_dates=True)
        return load_ff3(region_file)


def list_bundles() -> list[Path]:
    d = ROOT / "data"
    return sorted(p for p in d.iterdir() if (p / "prices.csv").exists()) if d.exists() else []


def read_wide_csv(file) -> pd.DataFrame:
    """Read a wide CSV (date column first, one column per ticker)."""
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.sort_index().apply(pd.to_numeric, errors="coerce")


# ------------------------------------------------------------------- factors
FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
ROOT = Path(__file__).resolve().parent.parent
LOCAL_FACTOR_DIR = ROOT / "data" / "factors"  # drop files downloaded in the browser here


def read_ff_file(path: Path) -> pd.DataFrame:
    """Parse a French data-library file as downloaded (ZIP or the CSV inside it)."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            raw = z.read(z.namelist()[0]).decode("latin-1")
    else:
        raw = path.read_text(encoding="latin-1")
    return parse_ff_csv(raw)


def load_ff3(region_file: str, cache_dir: Path = CACHE_DIR, local_dir: Path = LOCAL_FACTOR_DIR) -> pd.DataFrame:
    """Daily Fama-French 3 factors (decimal, columns Mkt-RF, SMB, HML, RF).

    Order: cache → file in data/factors/ (manual browser download) → Kenneth French website.
    """
    import requests

    cache = cache_dir / "factors" / region_file.replace(".zip", ".csv")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    stem = region_file.replace("_CSV.zip", "")
    for cand in (local_dir / region_file, local_dir / f"{stem}.csv", local_dir / f"{stem}.CSV"):
        if cand.exists():
            df = read_ff_file(cand)
            df.to_csv(cache)
            return df
    try:
        r = requests.get(FF_BASE + region_file, timeout=60, headers=BROWSER_HEADERS)
        r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"{exc}. Alternative: {FF_BASE}{region_file} im Browser laden und nach "
                           f"{local_dir} legen") from exc
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
