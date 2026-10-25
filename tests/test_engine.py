import numpy as np
import pandas as pd
import pytest

from eventstudy.checks import check_config
from eventstudy.config import Event, StudyConfig
from eventstudy.data import FrameProvider, parse_ff_csv
from eventstudy.engine import ALL, MAIN, run_study
from eventstudy.report import results_table, to_excel, to_markdown
from eventstudy.significance import apply_outliers


def row(res, group, window, model="market", variant=MAIN, sub=ALL):
    s = res.summary
    return s[(s["Group"] == group) & (s["Window"] == window) & (s["Model"] == model)
             & (s["Variant"] == variant) & (s["Subgroup"] == sub)].iloc[0]


def test_detects_injected_shock(synthetic, config):
    prices, _ = synthetic
    config.placebo.n = 30
    res = run_study(config, provider=FrameProvider(prices=prices))
    t = row(res, "Treated", "(0,0)")
    assert t["N"] == 39  # T39 excluded for low coverage
    assert t["CAAR"] == pytest.approx(-0.03, abs=0.006)
    for p in ("p_t", "p_wilcoxon", "p_patell", "p_bmp", "p_kolari", "p_rank", "p_sign"):
        assert t[p] < 0.01, p
    c = row(res, "Control", "(0,0)")
    assert abs(c["CAAR"]) < 0.006 and c["p_t"] > 0.01
    # subgroups are reported separately
    assert row(res, "Treated", "(0,0)", sub="A")["N"] == 20
    # treatment vs. control difference is significant
    d = res.differences[res.differences["Window"] == "(0,0)"].iloc[0]
    assert d["p_welch"] < 0.001
    # placebo: real event is extreme relative to random days
    p = res.placebo[res.placebo["Window"] == "(0,0)"].iloc[0]
    assert p["Percentile"] == 0 and p["p_empirisch"] < 0.05
    assert any("T39.DE" in e.message for e in res.protocol.exclusions())


def test_robustness_models(synthetic, config):
    prices, factors = synthetic
    config.robustness_models = ["ff3", "market_adjusted", "mean_adjusted"]
    config.placebo.enabled = False
    res = run_study(config, provider=FrameProvider(prices=prices), factor_loader=lambda _: factors)
    for m in ("ff3", "market_adjusted", "mean_adjusted"):
        assert row(res, "Treated", "(0,0)", model=m)["CAAR"] == pytest.approx(-0.03, abs=0.008), m
    assert "Winsorisiert 1 %" in set(res.summary["Variant"])


def test_ff3_missing_is_skipped_not_mocked(synthetic, config):
    prices, _ = synthetic
    config.robustness_models = ["ff3"]
    config.placebo.enabled = False

    def fail(_):
        raise ConnectionError("offline")

    res = run_study(config, provider=FrameProvider(prices=prices), factor_loader=fail)
    assert "ff3" not in set(res.summary["Model"])
    assert any("FF3" in w.message for w in res.protocol.warnings())


def test_reports(synthetic, config):
    prices, _ = synthetic
    config.placebo.enabled = False
    res = run_study(config, provider=FrameProvider(prices=prices))
    tbl = results_table(res)
    assert "***" in tbl[tbl["Gruppe"] == "Treated"]["CAAR"].iloc[0]
    assert to_excel(res)[:2] == b"PK"
    md = to_markdown(res)
    assert "## Methodik-Protokoll" in md and "Treated" in md


def test_full_event_window_requirement(synthetic, config):
    prices, _ = synthetic
    prices = prices.copy()
    e = prices.index.searchsorted(pd.Timestamp("2024-03-13"))
    prices.iloc[e + 2, prices.columns.get_loc("T0.DE")] = np.nan
    config.placebo.enabled = False
    res = run_study(config, provider=FrameProvider(prices=prices))
    assert row(res, "Treated", "(0,+5)")["N"] == 38
    assert row(res, "Treated", "(0,0)")["N"] == 39
    config.require_full_event_window = False
    res = run_study(config, provider=FrameProvider(prices=prices))
    assert row(res, "Treated", "(0,+5)")["N"] == 39


def test_outliers():
    cars = pd.DataFrame({"car": np.r_[np.zeros(98), 5.0, -5.0], "scar": np.r_[np.zeros(98), 9.0, -9.0]})
    w, aff = apply_outliers(cars, "winsorize", 0.01)
    assert len(w) == 100 and w["car"].abs().max() < 5 and len(aff) == 2
    t, aff = apply_outliers(cars, "trim", 0.01)
    assert len(t) == 98


def test_config_roundtrip(tmp_path, config):
    p = tmp_path / "c.json"
    config.save(p)
    back = StudyConfig.load(p)
    assert back.to_dict() == config.to_dict()
    assert back.windows[0] == (0, 0)


def test_checks(config):
    assert not [m for lvl, m in check_config(config) if lvl == "error"]
    config.estimation_window = (-250, 0)
    assert any(lvl == "error" for lvl, _ in check_config(config))
    config.estimation_window = (-250, -11)
    config.events.append(Event("2024-03-15", "zu nah"))
    assert any("überlappen" in m for _, m in check_config(config))


def test_parse_ff_csv():
    raw = "header\n\n,Mkt-RF,SMB,HML,RF\n20240102,  -1.00,  0.50, 0.10, 0.02\n20240103,0.20,0.1,0.0,0.02\n\nAnnual\n"
    df = parse_ff_csv(raw)
    assert len(df) == 2 and df.loc["2024-01-02", "Mkt-RF"] == pytest.approx(-0.01)
