import pandas as pd

from eventstudy.finder import Direction, catalog_ideas, confounders, idea_to_config, load_catalog
from eventstudy.screener import UniverseQuery, YahooScreener, expand_regions, industries_by_sector, \
    suggest_index_for

QUOTES = [
    {"symbol": "IFX.DE", "longName": "Infineon", "marketCap": 4e10, "currency": "EUR", "exchange": "GER"},
    {"symbol": "APC.DE", "longName": "Apple Inc.", "marketCap": 3e12, "currency": "EUR", "exchange": "GER"},
    {"symbol": "ASML.AS", "longName": "ASML", "marketCap": 3e11, "currency": "EUR", "exchange": "AMS"},
    {"symbol": "XYZ.DE", "longName": "Fonds", "quoteType": "ETF", "exchange": "GER"},
]
INFO = {"IFX.DE": {"country": "Germany", "industry": "Semiconductors"},
        "APC.DE": {"country": "United States", "industry": "Consumer Electronics"},
        "ASML.AS": {"country": "Netherlands", "industry": "Semiconductor Equipment & Materials"}}


def make_screener(tmp_path, calls):
    def page(query, offset, size):
        # like Yahoo: an industry filter only returns that industry's listings
        calls.append(offset)
        q = str(query)
        hits = [x for x in QUOTES if "industry" not in q or f"'{INFO.get(x['symbol'], {}).get('industry')}'" in q
                or x["symbol"] == "APC.DE"]
        return {"quotes": hits if offset == 0 else [], "total": len(hits)}

    return YahooScreener(cache_dir=tmp_path, fetch_page=page, fetch_info=lambda t: INFO.get(t, {}))


def test_screener_checks_headquarters_and_caches(tmp_path):
    calls = []
    sc = make_screener(tmp_path, calls)
    q = UniverseQuery(["EU"], industries=["Semiconductors", "Semiconductor Equipment & Materials"])
    df = sc.search(q)
    assert list(df["ticker"]) == ["ASML.AS", "IFX.DE"]  # sorted by market cap, Apple's Xetra line dropped
    assert set(df["country"]) == {"Deutschland", "Niederlande"}
    assert df.set_index("ticker").loc["ASML.AS", "industry"] == "Semiconductor Equipment & Materials"
    assert any(t == "APC.DE" for t, _ in sc.dropped)
    n_calls = len(calls)
    sc.search(q)  # second time from cache
    assert len(calls) == n_calls


def test_query_helpers():
    assert "Deutschland" in expand_regions(["EU"]) and "USA" not in expand_regions(["EU"])
    assert suggest_index_for(["Deutschland"]) == ("^GDAXI", "Europe")
    assert suggest_index_for(["Deutschland", "Frankreich"]) == ("^STOXX", "Europe")
    assert suggest_index_for(["USA"]) == ("^GSPC", "US")


def test_catalog_industries_are_valid_yahoo_industries():
    valid = {i for v in industries_by_sector().values() for i in v}
    for ev in load_catalog():
        assert set(ev.get("industries", [])) <= valid, ev["label"]
        assert pd.Timestamp(ev["date"]) <= pd.Timestamp.today()


def test_catalog_ideas_follow_direction():
    d = Direction(sectors=["Technology"], industries=["Semiconductors"], regions=["EU"], start="2020-01-01")
    ideas = catalog_ideas(d)
    labels = [i.events[0]["label"] for i in ideas]
    assert "DeepSeek-Schock" in labels and labels[0] == "DeepSeek-Schock"  # industry match ranks first
    assert all(i.events[0]["date"] >= "2020-01-01" for i in ideas)
    assert "Lehman" not in " ".join(labels)
    roles = [g.role for g in ideas[0].groups]
    assert roles == ["treatment", "control", "control"]
    assert ideas[0].groups[1].query.countries == ["USA"]  # same industry, other region


def test_event_type_filter_and_confounders():
    d = Direction(regions=["Deutschland"], event_types=["Handel & Zölle"], start="2025-01-01")
    ideas = catalog_ideas(d)
    assert ideas and all(i.event_type == "Handel & Zölle" for i in ideas)
    pause = next(i for i in ideas if i.events[0]["date"] == "2025-04-09")
    assert any("Liberation Day" in c for c in confounders(pause, load_catalog()))


def test_idea_to_config(tmp_path):
    d = Direction(sectors=["Technology"], industries=["Semiconductors", "Semiconductor Equipment & Materials"],
                  regions=["EU"], start="2024-01-01")
    idea = catalog_ideas(d)[0]
    sc = make_screener(tmp_path, [])
    universes = {idea.groups[0].name: sc.search(idea.groups[0].query)}
    cfg = idea_to_config(idea, universes, d)
    assert len(cfg.groups) == 1 and cfg.groups[0].role == "treatment"
    g = cfg.groups[0]
    assert g.market_index == "^STOXX" and g.subgroups  # two industries -> subgroups
    assert g.meta["IFX.DE"]["country"] == "Deutschland"
    assert "nicht angelegt" in cfg.description and cfg.placebo.groups == [g.name]
    assert cfg.events[0].date == idea.events[0]["date"]
