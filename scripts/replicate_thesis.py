"""Reproduce the bachelor thesis results with the tool's engine.

Uses the cleaned return files from the thesis repository
(https://github.com/ferdi-lil/EU-AI-Act-Event-Study, folder Data/) so the comparison
does not depend on Yahoo Finance data revisions.

    python scripts/replicate_thesis.py /path/to/EU-AI-Act-Event-Study/Data [--save]

--save stores the run as a study so it shows up in the app's dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eventstudy.config import StudyConfig  # noqa: E402
from eventstudy.data import FrameProvider, read_wide_csv  # noqa: E402
from eventstudy.engine import MAIN, run_study  # noqa: E402

INDEX_COLUMNS = {"STOXX600": "^STOXX", "SP500": "^GSPC", "FTSE100": "^FTSE"}
FILES = {"EU-Tech-Main": "returns_EU_Tech.csv", "EU-Non-Tech": "returns_EU_NonTech.csv",
         "US-Tech": "returns_US_Tech.csv", "UK-Tech": "returns_UK_Tech.csv"}

# Tables 1 / 2 of the thesis (market model): (event, group, window) -> CAAR in %, t statistic
M, D = "2024-03-13", "2023-12-09"
THESIS = {
    (D, "EU-Tech-Main", "(-1,+1)"): (0.01, 0.0285), (D, "EU-Tech-Main", "(0,+5)"): (0.72, 0.8799),
    (D, "EU-Non-Tech", "(0,+5)"): (-1.10, -2.4858), (D, "UK-Tech", "(-1,+1)"): (1.49, 2.4319),
    (M, "EU-Tech-Main", "(-1,+1)"): (-1.40, -2.2674), (M, "EU-Tech-Main", "(0,+1)"): (-1.54, -3.6208),
    (M, "EU-Tech-Main", "(0,+3)"): (-2.75, -4.6887), (M, "EU-Tech-Main", "(0,+5)"): (-4.14, -5.9616),
    (M, "EU-Tech-Main", "(-5,+5)"): (-6.89, -5.3448),
    (M, "EU-Non-Tech", "(-1,+1)"): (0.10, 0.3209), (M, "EU-Non-Tech", "(0,+3)"): (0.52, 1.4837),
    # US-Tech: the PDF (N=186) stems from an earlier data pull; the repository's Data/ and
    # Output/caar_summary.csv hold N=185, which is what is compared here.
    (M, "US-Tech", "(-1,+1)"): (-3.13, -9.9291), (M, "US-Tech", "(0,+3)"): (-2.13, -7.8995),
    (M, "UK-Tech", "(-1,+1)"): (-0.66, -0.9747), (M, "UK-Tech", "(0,+3)"): (-2.54, -2.9374),
}


def main(data_dir: Path) -> int:
    cfg = StudyConfig.load(ROOT / "templates" / "eu_ai_act.json")
    cfg.groups = [g for g in cfg.groups if g.name in FILES]
    frames = []
    for g in cfg.groups:
        df = read_wide_csv(data_dir / FILES[g.name])
        g.tickers = list(df.columns)  # the cleaned thesis sample
        frames.append(df)
    market = read_wide_csv(data_dir / "market_returns.csv").rename(columns=INDEX_COLUMNS)
    returns = pd.concat(frames + [market], axis=1)
    returns = returns.loc[:, ~returns.columns.duplicated()]

    cfg.name = "EU AI Act – Replikation (Thesis-Daten)"
    cfg.robustness_models = []  # FF3 factors need network access
    cfg.require_full_event_window = False  # thesis code summed available ARs
    cfg.min_coverage = 0.0  # files are already cleaned
    cfg.extreme_return_threshold = None  # thesis kept all returns (e.g. SQNS -65 % on 2024-02-23)
    cfg.outlier_robustness = False
    cfg.placebo.groups = ["EU-Tech-Main"]

    res = run_study(cfg, provider=FrameProvider(returns=returns, name="Thesis-Daten"))
    s = res.summary[(res.summary["Variant"] == MAIN) & (res.summary["Subgroup"] == "Alle")]
    ok = True
    print(f"{'Event':<12}{'Gruppe':<14}{'Fenster':<9}{'CAAR Tool':>10}{'CAAR BA':>9}{'t Tool':>9}{'t BA':>9}")
    for (event, group, window), (caar, t) in THESIS.items():
        row = s[(s["Event"] == event) & (s["Group"] == group) & (s["Window"] == window)].iloc[0]
        match = abs(row["CAAR"] * 100 - caar) < 0.006 and abs(row["t_stat"] - t) < 0.002
        ok &= match
        print(f"{event}  {group:<14}{window:<9}{row['CAAR'] * 100:>10.2f}{caar:>9.2f}{row['t_stat']:>9.4f}{t:>9.4f}  "
              f"{'✓' if match else '✗'}")
    print("\nZusätzliche Tests (EU-Tech-Main, 13.03.2024):")
    cols = ["Window", "N", "CAAR", "p_t", "p_wilcoxon", "p_bmp", "p_kolari", "rbar", "p_rank"]
    print(s[(s["Group"] == "EU-Tech-Main") & (s["Event"] == M)][cols].to_string(index=False))
    print("\nPlacebo (vgl. Tabelle 5):")
    print(res.placebo.to_string(index=False))
    if "--save" in sys.argv:
        from eventstudy import storage

        slug = storage.save_config(cfg)
        storage.save_result(slug, res)
        print(f"\nGespeichert als Studie '{slug}' – im Dashboard sichtbar.")
    return 0 if ok else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(Path(args[0]) if args else Path("../EU-AI-Act-Event-Study/Data")))
