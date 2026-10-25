"""Company universe for guided group selection.

A universe is a table with at least `ticker`; optional columns: name, country, region,
industry, sector, market_cap. Users can upload their own screening export (e.g. from
S&P Capital IQ, Refinitiv, Bloomberg) and map its columns.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BUILTIN = ROOT / "universes"

STANDARD_COLUMNS = ["ticker", "name", "country", "region", "industry", "sector", "market_cap"]

# Ticker suffix -> country (Yahoo Finance conventions)
SUFFIX_COUNTRY = {
    "DE": "Deutschland", "F": "Deutschland", "PA": "Frankreich", "AS": "Niederlande", "BR": "Belgien",
    "MI": "Italien", "MC": "Spanien", "LS": "Portugal", "VI": "Österreich", "SW": "Schweiz",
    "ST": "Schweden", "CO": "Dänemark", "HE": "Finnland", "OL": "Norwegen", "WA": "Polen",
    "IR": "Irland", "AT": "Griechenland", "PR": "Tschechien", "L": "Vereinigtes Königreich",
    "TO": "Kanada", "AX": "Australien", "T": "Japan", "HK": "Hongkong",
}
EU_COUNTRIES = {"Deutschland", "Frankreich", "Niederlande", "Belgien", "Italien", "Spanien", "Portugal",
                "Österreich", "Schweden", "Dänemark", "Finnland", "Polen", "Irland", "Griechenland",
                "Tschechien", "Luxemburg"}


def guess_country(ticker: str) -> str:
    if "." in ticker:
        return SUFFIX_COUNTRY.get(ticker.rsplit(".", 1)[1].upper(), "")
    return "" if ticker.startswith("^") else "USA (oder US-Listing)"


def region_of(country: str) -> str:
    if country in EU_COUNTRIES:
        return "EU"
    if country == "Vereinigtes Königreich":
        return "UK"
    if country.startswith("USA"):
        return "USA"
    return "Sonstige" if country else ""


def suggest_index(tickers: list[str]) -> str:
    """Most common region of the tickers -> a broad market index."""
    regions = pd.Series([region_of(guess_country(t)) for t in tickers]).value_counts()
    top = regions.index[0] if len(regions) else "EU"
    return {"EU": "^STOXX", "UK": "^FTSE", "USA": "^GSPC"}.get(top, "URTH")


def list_builtin() -> list[Path]:
    return sorted(BUILTIN.glob("*.csv"))


def load_universe(source, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
    if mapping:
        df = df.rename(columns={v: k for k, v in mapping.items() if v})
    df = df[[c for c in STANDARD_COLUMNS if c in df.columns] + [c for c in df.columns if c not in STANDARD_COLUMNS]]
    if "ticker" not in df.columns:
        raise ValueError("Die Universumsdatei braucht eine Ticker-Spalte")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if "country" not in df.columns:
        df["country"] = df["ticker"].map(guess_country)
    if "region" not in df.columns:
        df["region"] = df["country"].fillna("").map(region_of)
    if "market_cap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    return df.drop_duplicates("ticker")


def categorical_columns(df: pd.DataFrame, max_levels: int = 300) -> list[str]:
    """Columns that make sense as filters (few distinct text values)."""
    cols = []
    for c in df.columns:
        if c in ("ticker", "name") or pd.api.types.is_numeric_dtype(df[c]):
            continue
        if 1 < df[c].nunique() <= max_levels:
            cols.append(c)
    return cols


def filter_universe(df: pd.DataFrame, filters: dict[str, list] | None = None, min_cap: float | None = None,
                    top_n: int | None = None, top_n_by: list[str] | None = None) -> pd.DataFrame:
    """Keep rows whose columns match the selected values; optionally the N largest per category."""
    out = df
    for col, values in (filters or {}).items():
        if values:
            out = out[out[col].isin(values)]
    if min_cap and "market_cap" in out.columns:
        out = out[out["market_cap"] >= min_cap]
    if top_n and "market_cap" in out.columns:
        keys = [c for c in (top_n_by or []) if c in out.columns]
        out = out.sort_values("market_cap", ascending=False)
        out = out.groupby(keys, group_keys=False).head(top_n) if keys else out.head(top_n)
    return out


def parse_ticker_text(text: str) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Parse lines like `SAP.DE`, `SAP.DE; SAP SE` or `SAP.DE; SAP SE; High`."""
    tickers, names, subgroups = [], {}, {}
    for line in text.replace(",", "\n").splitlines() if ";" not in text else text.splitlines():
        parts = [p.strip() for p in line.split(";")]
        if not parts or not parts[0]:
            continue
        t = parts[0].upper() if not parts[0].startswith("^") else parts[0]
        if t in tickers:
            continue
        tickers.append(t)
        if len(parts) > 1 and parts[1]:
            names[t] = parts[1]
        if len(parts) > 2 and parts[2]:
            subgroups[t] = parts[2]
    return tickers, names, subgroups
