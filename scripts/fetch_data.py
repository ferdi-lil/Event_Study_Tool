"""Download all data a study needs into a data bundle (data/<name>/).

Run this on a machine with internet access; the app (or a cloud session without access to
Yahoo) can then use the bundle via "Datenpaket" in step 6 – the same idea as loading the data
in R and reading the Excel file afterwards.

    python scripts/fetch_data.py templates/eu_ai_act.json
    python scripts/fetch_data.py studies/<slug>/config.json --name meine-daten

Output: prices.csv (adjusted close, one column per ticker incl. indices), sources.csv
(where each ticker came from), ff3_<region>.csv for every FF3 region, manifest.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eventstudy.config import FF3_REGIONS, StudyConfig  # noqa: E402
from eventstudy.data import YahooProvider, load_ff3  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="Studien-Konfiguration (JSON)")
    ap.add_argument("--name", help="Name des Datenpakets (Standard: Name der Studie)")
    ap.add_argument("--no-stooq", action="store_true", help="Stooq nicht als Notlösung verwenden")
    args = ap.parse_args()

    cfg = StudyConfig.load(Path(args.config))
    out = ROOT / "data" / (args.name or cfg.slug)
    out.mkdir(parents=True, exist_ok=True)
    start, end = cfg.data_period()
    tickers = sorted({t for g in cfg.groups for t in g.tickers} | {g.market_index for g in cfg.groups})
    print(f"Lade {len(tickers)} Ticker von {start} bis {end} …")

    provider = YahooProvider(use_stooq=not args.no_stooq)
    prices = provider.prices(tickers, start, end,
                             progress=lambda f, text: print(f"\r  {text:<70}", end="", flush=True))
    print()
    if prices.empty:
        print("Keine Kurse erhalten. Beispiel-Fehler:", next(iter(provider.errors.values()), "–"))
        return 1
    prices.to_csv(out / "prices.csv")
    pd.Series(provider.sources, name="source").rename_axis("ticker").to_csv(out / "sources.csv")
    missing = [t for t in tickers if t not in prices.columns]

    factors = {}
    if "ff3" in cfg.all_models():
        for region in sorted({g.ff3_region for g in cfg.groups}):
            file = FF3_REGIONS[region]
            try:
                load_ff3(file).to_csv(out / f"ff3_{file.replace('_CSV.zip', '')}.csv")
                factors[region] = "ok"
            except Exception as exc:
                factors[region] = f"fehlgeschlagen: {exc}"

    counts = pd.Series(provider.sources).value_counts().to_dict()
    manifest = {"study": cfg.name, "created": datetime.now().isoformat(timespec="seconds"),
                "period": [str(start), str(end)], "tickers": len(tickers), "loaded": prices.shape[1],
                "sources": counts, "missing": missing, "ff3": factors}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDatenpaket: {out}")
    print(f"  {prices.shape[1]}/{len(tickers)} Ticker geladen – Quellen: {counts}")
    if missing:
        print(f"  Ohne Daten ({len(missing)}, meist delistet/übernommen oder später an die Börse gegangen):")
        print("   ", ", ".join(missing))
    for region, status in factors.items():
        print(f"  FF3 {region}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
