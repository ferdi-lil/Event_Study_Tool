import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from eventstudy import data as data_mod
from eventstudy.data import BundleProvider, YahooProvider, load_ff3, parse_yahoo_chart, stooq_symbol
from eventstudy.engine import run_study

FF_RAW = ("This file was created ...\n\n,Mkt-RF,SMB,HML,RF\n"
          "20240311,  -0.50,  0.10, 0.20, 0.02\n20240312,0.30,-0.1,0.0,0.02\n")


def test_parse_yahoo_chart_uses_adjclose_and_exchange_date():
    # Yahoo stamps daily bars at the session open in UTC; gmtoffset shifts them to exchange time
    t1 = int(pd.Timestamp("2024-03-11 23:30", tz="UTC").timestamp())  # = 00:30 on 12.03. in Frankfurt
    t2 = int(pd.Timestamp("2024-03-13 08:00", tz="UTC").timestamp())
    payload = {"chart": {"result": [{
        "meta": {"gmtoffset": 3600},
        "timestamp": [t1, t2],
        "indicators": {"quote": [{"close": [100.0, 101.0]}], "adjclose": [{"adjclose": [99.0, None]}]},
    }]}}
    s = parse_yahoo_chart(payload)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-03-12"]
    assert s.iloc[0] == 99.0


def test_parse_yahoo_chart_empty():
    with pytest.raises(ValueError):
        parse_yahoo_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})


def test_stooq_symbols():
    assert stooq_symbol("SAP.DE") == "sap.de"
    assert stooq_symbol("MSFT") == "msft.us"
    assert stooq_symbol("^GSPC") == "^spx"
    assert stooq_symbol("^STOXX") is None


def test_fallback_chain(tmp_path, monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=5)
    monkeypatch.setattr(YahooProvider, "_yfinance", lambda self, t, s, e: {"A.DE": pd.Series(1.0, idx)})

    def chart(t, s, e, sess=None):
        if t == "B.DE":
            return pd.Series(2.0, idx)
        raise RuntimeError("blocked")

    def stooq(t, s, e, sess=None):
        if t == "C.DE":
            return pd.Series(3.0, idx)
        raise ValueError("no data")

    monkeypatch.setattr(data_mod, "fetch_yahoo_chart", chart)
    monkeypatch.setattr(data_mod, "fetch_stooq", stooq)
    p = YahooProvider(cache_dir=tmp_path)
    df = p.prices(["A.DE", "B.DE", "C.DE", "D.DE"], "2024-01-01", "2024-01-10")
    assert list(df.columns) == ["A.DE", "B.DE", "C.DE"]
    assert p.sources == {"A.DE": "yfinance", "B.DE": "Yahoo-Chart-API", "C.DE": "Stooq (unbereinigt)"}
    assert "D.DE" in p.errors
    # second call is served from cache and keeps the original source
    p2 = YahooProvider(cache_dir=tmp_path)
    monkeypatch.setattr(YahooProvider, "_yfinance", lambda self, t, s, e: {})
    df2 = p2.prices(["B.DE", "C.DE"], "2024-01-01", "2024-01-10")
    assert df2.shape == (5, 2) and p2.sources["C.DE"].startswith("Stooq")


def test_ff3_from_local_browser_download(tmp_path):
    local = tmp_path / "factors"
    local.mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Europe_3_Factors_Daily.csv", FF_RAW)
    (local / "Europe_3_Factors_Daily_CSV.zip").write_bytes(buf.getvalue())
    df = load_ff3("Europe_3_Factors_Daily_CSV.zip", cache_dir=tmp_path / "cache", local_dir=local)
    assert df.loc["2024-03-11", "Mkt-RF"] == pytest.approx(-0.005)


def test_study_from_bundle(tmp_path, synthetic, config):
    prices, factors = synthetic
    prices.to_csv(tmp_path / "prices.csv")
    pd.Series({"T0.DE": "yfinance", "T1.DE": "Stooq (unbereinigt)"}, name="source").rename_axis("ticker") \
        .to_csv(tmp_path / "sources.csv")
    factors.to_csv(tmp_path / "ff3_Europe_3_Factors_Daily.csv")
    config.robustness_models = ["ff3"]
    config.placebo.enabled = False
    bp = BundleProvider(tmp_path)
    res = run_study(config, provider=bp, factor_loader=bp.factor_loader)
    assert "ff3" in set(res.summary["Model"])
    assert any("Stooq" in w.message and "T1.DE" in w.message for w in res.protocol.warnings())
    assert np.isfinite(res.main_table()["CAAR"]).all()
