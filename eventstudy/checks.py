"""Methodology checks shown in the wizard before a study is run."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from .config import StudyConfig


def check_config(cfg: StudyConfig) -> list[tuple[str, str]]:
    """Return (level, message) with level in error | warning | hint."""
    out: list[tuple[str, str]] = []
    if not cfg.events:
        out.append(("error", "Mindestens ein Event-Datum angeben."))
    if not cfg.treatment_groups():
        out.append(("error", "Mindestens eine Untersuchungsgruppe anlegen."))
    if not cfg.windows:
        out.append(("error", "Mindestens ein Eventfenster wählen."))
    if out:
        return out

    est0, est1 = cfg.estimation_window
    lo = min(w[0] for w in cfg.windows)
    hi = max(w[1] for w in cfg.windows)
    if est1 >= lo:
        out.append(("error", f"Schätzfenster endet bei {est1}, Eventfenster beginnt bei {lo}: die Fenster überlappen."))
    if est1 - est0 + 1 < 100:
        out.append(("warning", "Schätzfenster unter 100 Handelstagen – Parameter werden unpräzise geschätzt."))
    if cfg.min_estimation_obs > est1 - est0 + 1:
        out.append(("error", "Mindestbeobachtungen sind größer als das Schätzfenster."))

    today = pd.Timestamp.today().normalize()
    events = sorted(cfg.events, key=lambda e: e.date)
    for ev in events:
        ts = ev.timestamp
        if ts.day_of_week >= 5:
            out.append(("hint", f"{ev.date} ist ein Wochenende – t=0 wird der nächste Handelstag."))
        if ts + pd.tseries.offsets.BDay(hi + 1) > today:
            out.append(("error", f"Für {ev.date} liegen noch nicht genug Handelstage nach dem Event vor (bis +{hi})."))
    for a, b in zip(events, events[1:]):
        gap = len(pd.bdate_range(a.timestamp, b.timestamp)) - 1
        if gap <= hi - lo:
            out.append(("warning", f"Eventfenster von {a.date} und {b.date} überlappen ({gap} Handelstage Abstand). "
                                   "Die Effekte lassen sich dann nicht sauber trennen."))
        elif gap < -est0:
            out.append(("warning", f"Das Schätzfenster von {b.date} enthält das Event {a.date}. "
                                   "Die Normalrendite kann dadurch verzerrt sein (ggf. Schätzfenster verkürzen)."))
    if hi - lo + 1 > 21:
        out.append(("warning", "Eventfenster über 20 Tage erhöhen das Risiko konfundierender Ereignisse stark."))

    for g in cfg.groups:
        n = len(g.tickers)
        if n == 0:
            out.append(("error", f"Gruppe „{g.name}“ enthält keine Unternehmen."))
        elif n < 10:
            out.append(("warning", f"Gruppe „{g.name}“ hat nur {n} Unternehmen – Tests haben wenig Aussagekraft."))
        elif n < 30:
            out.append(("hint", f"Gruppe „{g.name}“ hat {n} Unternehmen – nicht-parametrische Tests besonders beachten."))
    treated = {t for g in cfg.treatment_groups() for t in g.tickers}
    overlap = sorted(treated & {t for g in cfg.control_groups() for t in g.tickers})
    if overlap:
        out.append(("warning", f"{len(overlap)} Ticker sind zugleich in Untersuchungs- und Kontrollgruppe: "
                               f"{', '.join(overlap[:8])}"))
    nested = [t for t, c in Counter(t for g in cfg.treatment_groups() for t in g.tickers).items() if c > 1]
    if nested:
        out.append(("hint", f"{len(nested)} Ticker stehen in mehreren Untersuchungsgruppen (verschachtelte "
                            "Stichproben, z. B. Haupt- und erweiterte Gruppe) – das ist zulässig."))

    if not cfg.control_groups():
        out.append(("hint", "Keine Kontrollgruppe: Ohne Vergleich lässt sich ein allgemeiner Markteffekt kaum "
                            "von einem ereignisspezifischen Effekt unterscheiden."))
    if "kolari" not in cfg.tests and "bmp" not in cfg.tests:
        out.append(("hint", "Alle Unternehmen haben dasselbe Event-Datum (Event-Clustering). Der einfache t-Test "
                            "überschätzt dann oft die Signifikanz – Kolari-Pynnönen-Test empfohlen."))
    if "ff3" in cfg.all_models() and cfg.return_type == "log":
        out.append(("hint", "FF3-Faktoren sind einfache Renditen; mit Log-Renditen ist das eine (übliche) Näherung."))
    if not cfg.placebo.enabled:
        out.append(("hint", "Placebo-Tests sind deaktiviert. Sie zeigen, ob der Effekt ungewöhnlich gegenüber "
                            "zufälligen Tagen ist."))
    return out
