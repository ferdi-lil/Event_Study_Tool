"""On-demand company universes: which listed companies belong to sector X in region Y?

Nothing is downloaded in advance. A query runs Yahoo's equity screener (listing exchange,
sector/industry, market cap), then checks each hit's headquarters country, because primary
exchanges also list foreign shares (Xetra trades Apple as APC.DE). Results are cached.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .data import CACHE_DIR

# Country label -> (Yahoo screener region, primary exchange codes, Yahoo HQ country name)
COUNTRIES = {
    "Deutschland": ("de", ["GER"], "Germany"),
    "Frankreich": ("fr", ["PAR"], "France"),
    "Niederlande": ("nl", ["AMS"], "Netherlands"),
    "Belgien": ("be", ["BRU"], "Belgium"),
    "Italien": ("it", ["MIL"], "Italy"),
    "Spanien": ("es", ["MCE"], "Spain"),
    "Portugal": ("pt", ["LIS"], "Portugal"),
    "Österreich": ("at", ["VIE"], "Austria"),
    "Irland": ("ie", ["ISE"], "Ireland"),
    "Finnland": ("fi", ["HEL"], "Finland"),
    "Schweden": ("se", ["STO"], "Sweden"),
    "Dänemark": ("dk", ["CPH"], "Denmark"),
    "Polen": ("pl", ["WSE"], "Poland"),
    "Griechenland": ("gr", ["ATH"], "Greece"),
    "Norwegen": ("no", ["OSL"], "Norway"),
    "Schweiz": ("ch", ["EBS"], "Switzerland"),
    "Vereinigtes Königreich": ("gb", ["LSE"], "United Kingdom"),
    "USA": ("us", ["NMS", "NYQ", "NGM", "NCM", "ASE"], "United States"),
    "Japan": ("jp", ["JPX"], "Japan"),
}
EU = ["Deutschland", "Frankreich", "Niederlande", "Belgien", "Italien", "Spanien", "Portugal", "Österreich",
      "Irland", "Finnland", "Schweden", "Dänemark", "Polen", "Griechenland"]
REGIONS = {
    "EU": EU,
    "Europa (EU + UK, CH, NO)": EU + ["Vereinigtes Königreich", "Schweiz", "Norwegen"],
    "USA": ["USA"],
}

SECTORS = {
    "Technology": "Technologie", "Communication Services": "Kommunikation & Medien",
    "Consumer Cyclical": "Zyklischer Konsum", "Consumer Defensive": "Basiskonsum",
    "Energy": "Energie", "Financial Services": "Finanzen", "Healthcare": "Gesundheit",
    "Industrials": "Industrie", "Basic Materials": "Grundstoffe", "Real Estate": "Immobilien",
    "Utilities": "Versorger",
}


def industries_by_sector() -> dict[str, list[str]]:
    try:
        from yfinance.const import EQUITY_SCREENER_EQ_MAP

        return {s: sorted(v) for s, v in EQUITY_SCREENER_EQ_MAP["industry"].items()}
    except Exception:  # yfinance missing or changed – sectors still work
        return {s: [] for s in SECTORS}


def expand_regions(labels: list[str]) -> list[str]:
    """Region or country labels -> unique country labels."""
    out = []
    for lab in labels:
        for c in REGIONS.get(lab, [lab]):
            if c in COUNTRIES and c not in out:
                out.append(c)
    return out


def suggest_index_for(countries: list[str]) -> tuple[str, str]:
    """(market index ticker, FF3 region) that fit a set of HQ countries."""
    single = {"Deutschland": "^GDAXI", "Frankreich": "^FCHI", "Vereinigtes Königreich": "^FTSE",
              "Schweiz": "^SSMI", "USA": "^GSPC", "Japan": "^N225"}
    if len(countries) == 1 and countries[0] in single:
        idx = single[countries[0]]
    elif countries and all(c == "USA" for c in countries):
        idx = "^GSPC"
    else:
        idx = "^STOXX"
    ff3 = "US" if idx == "^GSPC" else "Japan" if idx == "^N225" else "Europe"
    return idx, ff3


@dataclass
class UniverseQuery:
    countries: list[str]  # labels from COUNTRIES (or regions, expanded on the fly)
    sectors: list[str] = field(default_factory=list)  # Yahoo sector names
    industries: list[str] = field(default_factory=list)  # Yahoo industry names (take precedence)
    exclude_sectors: list[str] = field(default_factory=list)
    min_market_cap_usd: float = 5e8
    max_results: int = 250

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    def describe(self) -> str:
        what = (", ".join(self.industries) if self.industries else
                ", ".join(SECTORS.get(s, s) for s in self.sectors) if self.sectors else "alle Sektoren")
        if self.exclude_sectors:
            what += " (ohne " + ", ".join(SECTORS.get(s, s) for s in self.exclude_sectors) + ")"
        return (f"Börsennotierte Unternehmen mit Sitz in {', '.join(expand_regions(self.countries))}; {what}; "
                f"Marktkapitalisierung mind. {self.min_market_cap_usd / 1e6:.0f} Mio. USD; "
                "Quelle: Yahoo-Finance-Screener, Sitzland je Unternehmen geprüft")


class YahooScreener:
    """`fetch_page(query_obj, offset, size)` and `fetch_info(ticker)` are injectable for tests."""

    name = "Yahoo-Finance-Screener"

    def __init__(self, cache_dir: Path = CACHE_DIR, fetch_page=None, fetch_info=None, ttl_days: int = 7) -> None:
        self.cache_dir = cache_dir / "universe"
        self.meta_dir = cache_dir / "meta"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.fetch_page = fetch_page or self._yahoo_page
        self.fetch_info = fetch_info or self._yahoo_info
        self.ttl = ttl_days * 86400
        self.dropped: list[tuple[str, str]] = []  # (ticker, reason) for the protocol

    # ---------------------------------------------------------------- yahoo
    @staticmethod
    def _quiet():
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        stack = contextlib.ExitStack()
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore")
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def _yahoo_page(self, query, offset: int, size: int) -> dict:
        import yfinance as yf

        with self._quiet():
            return yf.screen(query, offset=offset, size=size, sortField="intradaymarketcap", sortAsc=False)

    def _yahoo_info(self, ticker: str) -> dict:
        import yfinance as yf

        with self._quiet():
            info = yf.Ticker(ticker).info or {}
        return {k: info.get(k) for k in ("country", "sector", "industry", "marketCap", "currency", "longName",
                                         "shortName", "quoteType")}

    @staticmethod
    def build_query(q: UniverseQuery, countries: list[str], industry: str | None, sector: str | None):
        from yfinance import EquityQuery

        exchanges = [e for c in countries for e in COUNTRIES[c][1]]
        parts = [EquityQuery("is-in", ["exchange", *exchanges]),
                 EquityQuery("gte", ["intradaymarketcap", q.min_market_cap_usd])]
        if industry:
            parts.append(EquityQuery("eq", ["industry", industry]))
        elif sector:
            parts.append(EquityQuery("eq", ["sector", sector]))
        return EquityQuery("and", parts)

    # ---------------------------------------------------------------- cache
    def _cached(self, path: Path):
        return path.exists() and time.time() - path.stat().st_mtime < self.ttl

    def info(self, ticker: str) -> dict:
        p = self.meta_dir / f"{ticker.replace('/', '_')}.json"
        if self._cached(p):
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                pass
        data = self.fetch_info(ticker)
        p.write_text(json.dumps(data), encoding="utf-8")
        return data

    # ---------------------------------------------------------------- search
    def search(self, q: UniverseQuery, progress=None) -> pd.DataFrame:
        """Columns: ticker, name, country, sector, industry, market_cap, currency, exchange."""
        countries = expand_regions(q.countries)
        cache = self.cache_dir / f"{q.key()}.csv"
        if self._cached(cache):
            return pd.read_csv(cache)
        self.dropped = []

        def report(f, text):
            if progress:
                progress(f, text)

        # one screener pass per industry (or sector) so every hit carries its classification
        if q.industries:
            passes = [(None, ind) for ind in q.industries]
        elif q.sectors:
            passes = [(s, None) for s in q.sectors]
        else:
            passes = [(s, None) for s in SECTORS if s not in q.exclude_sectors]
        sector_of = {ind: s for s, inds in industries_by_sector().items() for ind in inds}
        rows = {}
        for i, (sector, industry) in enumerate(passes):
            report(0.4 * i / len(passes), f"Screener: {industry or SECTORS.get(sector, sector)} …")
            query = self.build_query(q, countries, industry, sector)
            offset = 0
            while offset < q.max_results:
                res = self.fetch_page(query, offset, min(250, q.max_results - offset)) or {}
                quotes = res.get("quotes") or []
                for qt in quotes:
                    t = qt.get("symbol")
                    if not t or t in rows or qt.get("quoteType", "EQUITY") != "EQUITY":
                        continue
                    rows[t] = {"ticker": t, "name": qt.get("longName") or qt.get("shortName") or "",
                               "sector": sector or sector_of.get(industry, ""), "industry": industry or "",
                               "market_cap": qt.get("marketCap"), "currency": qt.get("currency"),
                               "exchange": qt.get("exchange")}
                offset += len(quotes)
                if len(quotes) == 0 or offset >= (res.get("total") or 0):
                    break

        # headquarters check (parallel, cached per ticker) – only for the largest candidates, with
        # headroom for listings that turn out to be foreign
        wanted = {COUNTRIES[c][2]: c for c in countries}
        tickers = sorted(rows, key=lambda t: -(rows[t]["market_cap"] or 0))[:int(q.max_results * 1.5) + 10]
        infos = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for k, (t, inf) in enumerate(zip(tickers, ex.map(self._safe_info, tickers))):
                infos[t] = inf
                if k % 10 == 0:
                    report(0.4 + 0.6 * k / max(len(tickers), 1), f"Prüfe Firmensitz {k}/{len(tickers)} …")
        out = []
        for t in tickers:
            r, inf = rows[t], infos[t]
            hq = inf.get("country")
            if hq not in wanted:
                self.dropped.append((t, f"Sitz {hq or 'unbekannt'} – nur Börsennotiz in der Region"))
                continue
            if q.exclude_sectors and (inf.get("sector") or r["sector"]) in q.exclude_sectors:
                continue
            r["country"] = wanted[hq]
            r["industry"] = r["industry"] or inf.get("industry") or ""
            r["sector"] = r["sector"] or inf.get("sector") or ""
            r["name"] = r["name"] or inf.get("longName") or ""
            r["market_cap"] = r["market_cap"] or inf.get("marketCap")
            r["currency"] = r["currency"] or inf.get("currency")
            out.append(r)
        report(1.0, f"{len(out)} Unternehmen gefunden")
        df = pd.DataFrame(out, columns=["ticker", "name", "country", "sector", "industry", "market_cap",
                                        "currency", "exchange"])
        df = df.sort_values("market_cap", ascending=False, na_position="last").head(q.max_results)
        df = df.reset_index(drop=True)
        if not rows:
            raise RuntimeError("Der Screener hat keine Treffer geliefert (Verbindung oder Filter prüfen).")
        df.to_csv(cache, index=False)
        return df

    def _safe_info(self, ticker: str) -> dict:
        try:
            return self.info(ticker) or {}
        except Exception:
            return {}
