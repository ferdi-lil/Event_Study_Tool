import json
import platform
import time
from datetime import date, timedelta
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
import streamlit as st

from eventstudy import data as data_mod
from ui import style

style.titlebar("System", "Datenquellen prüfen · Zwischenspeicher verwalten")


def pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "nicht installiert"


style.section("Umgebung")
st.dataframe(pd.DataFrame([
    {"Komponente": "Python", "Version": platform.python_version()},
    {"Komponente": "yfinance", "Version": pkg("yfinance")},
    {"Komponente": "pandas", "Version": pkg("pandas")},
    {"Komponente": "streamlit", "Version": pkg("streamlit")},
    {"Komponente": "requests", "Version": pkg("requests")},
]), hide_index=True, use_container_width=False)

style.section("Datenquellen-Check", "Ruft jede Quelle mit einer kleinen Testabfrage ab (SAP, letzte 30 Tage). "
                                    "Dauert 10–60 Sekunden.")


def run_check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        return {"Quelle": name, "Status": "OK", "Dauer (s)": round(time.time() - t0, 1), "Details": detail}
    except Exception as exc:  # report every failure verbatim
        return {"Quelle": name, "Status": "Fehler", "Dauer (s)": round(time.time() - t0, 1),
                "Details": f"{type(exc).__name__}: {str(exc)[:300]}"}


def check_yfinance():
    p = data_mod.YahooProvider(cache_dir=data_mod.CACHE_DIR / "_check")
    end = date.today()
    got = p._yfinance(["SAP.DE"], end - timedelta(days=30), end)
    if "SAP.DE" not in got:
        raise RuntimeError(p.errors.get("SAP.DE", "keine Daten erhalten"))
    return f"{len(got['SAP.DE'])} Handelstage"


def check_chart():
    end = date.today()
    s = data_mod.fetch_yahoo_chart("SAP.DE", end - timedelta(days=30), end)
    return f"{len(s)} Handelstage"


def check_stooq():
    end = date.today()
    s = data_mod.fetch_stooq("SAP.DE", end - timedelta(days=30), end)
    return f"{len(s)} Handelstage (nicht dividendenbereinigt)"


def check_screener():
    from eventstudy.screener import UniverseQuery, YahooScreener

    q = YahooScreener.build_query(UniverseQuery(["Deutschland"], sectors=["Technology"]), ["Deutschland"],
                                  None, "Technology")
    res = YahooScreener(cache_dir=data_mod.CACHE_DIR / "_check")._yahoo_page(q, 0, 5) or {}
    quotes = res.get("quotes") or []
    if not quotes:
        raise RuntimeError("keine Treffer")
    return ", ".join(x.get("symbol", "?") for x in quotes)


def check_ff3():
    import requests

    r = requests.get(data_mod.FF_BASE + "Europe_3_Factors_Daily_CSV.zip", timeout=60,
                     headers=data_mod.BROWSER_HEADERS)
    r.raise_for_status()
    return f"{len(r.content) / 1e6:.1f} MB erhalten"


if st.button("Check starten", type="primary"):
    rows = []
    bar = st.progress(0.0)
    checks = [("yfinance (Kurse, Standardweg)", check_yfinance), ("Yahoo-Chart-API (Kurse, Ersatz 1)", check_chart),
              ("Stooq (Kurse, Ersatz 2)", check_stooq), ("Yahoo-Screener (Firmenlisten)", check_screener),
              ("Kenneth French (FF3-Faktoren)", check_ff3)]
    for i, (name, fn) in enumerate(checks):
        bar.progress(i / len(checks), f"Prüfe {name} …")
        rows.append(run_check(name, fn))
    bar.empty()
    st.session_state["system_check"] = rows

rows = st.session_state.get("system_check")
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    ok = {r["Quelle"].split(" (")[0]: r["Status"] == "OK" for r in rows}
    if not ok.get("yfinance") and not ok.get("Yahoo-Chart-API"):
        st.error("Keine Yahoo-Quelle erreichbar. Häufige Ursachen: keine Internetverbindung, Firmen-/Uni-Netz "
                 "mit Proxy, VPN, oder Yahoo drosselt gerade (dann in 15–60 Minuten erneut versuchen). "
                 "Eine veraltete yfinance-Version lässt sich im Terminal mit "
                 "`pip install -U yfinance` aktualisieren.")
    elif not ok.get("yfinance"):
        st.warning("yfinance scheitert, die Ersatzquelle funktioniert aber. Downloads laufen dann langsamer. "
                   "Abhilfe meist: `pip install -U yfinance` im Terminal (bei aktivierter .venv).")
    else:
        st.success("Kursdaten sind erreichbar.")

style.section("Zwischenspeicher")
prices_dir = data_mod.CACHE_DIR / "prices"
files = list(prices_dir.glob("*.csv")) if prices_dir.exists() else []
failures = {}
for f in prices_dir.glob("_nicht_verfuegbar_*.json") if prices_dir.exists() else []:
    try:
        failures.update(json.loads(f.read_text(encoding="utf-8")))
    except ValueError:
        pass
c1, c2 = st.columns(2)
c1.metric("Zwischengespeicherte Kursreihen", len(files))
c2.metric("Als „nicht verfügbar“ gemerkt (24 h)", len(failures))
if failures:
    with st.expander("Ticker ohne Daten anzeigen"):
        st.dataframe(pd.DataFrame([{"Ticker": t, "Grund": r} for t, r in failures.items()]), hide_index=True,
                     use_container_width=True)
a, b, _ = st.columns([1.4, 1.4, 3])
if a.button("„Nicht verfügbar“-Liste leeren", use_container_width=True):
    n = data_mod.clear_price_cache(only_failures=True)
    st.success(f"{n} Datei(en) gelöscht. Beim nächsten Laden werden alle Ticker erneut versucht.")
if b.button("Gesamten Kurs-Cache leeren", use_container_width=True):
    n = data_mod.clear_price_cache()
    st.success(f"{n} Datei(en) gelöscht. Kurse werden beim nächsten Mal neu geladen.")
