# Event-Study-Tool

Kapitalmarkt-Event-Studies zu beliebigen Ereignissen, von der Gruppenauswahl bis zur fertigen Auswertung.
Das Tool basiert auf dem Design der Bachelorarbeit *„Untersuchung der Aktienmarktreaktionen europäischer
Tech-Unternehmen auf den EU AI Act“* (F. Lilienthal, HU Berlin 2026) und verallgemeinert es auf jedes Ereignis.

## Funktionen

- **Dashboard** mit allen Studien (Entwürfe und ausgewertete Studien), Duplizieren und Bearbeiten
- **Assistent in 6 Schritten:** Events → Untersuchungsgruppe(n) → Kontrollgruppen → Fenster & Modell → Robustheit & Tests → Daten & Start
- **Gruppenauswahl** auf drei Wegen:
  - *manuell* (Ticker-Liste oder CSV)
  - *angeleitet* (Filter auf einem Unternehmensuniversum, z. B. einem Capital-IQ-Export: Region, Branche, Marktkapitalisierung, „N größte je Branche“)
  - *KI-gestützt* (Claude schlägt Kriterien und Unternehmen vor, du prüfst und übernimmst)
- **Renditemodelle:** Marktmodell, Fama-French-3-Faktoren, marktbereinigt, Constant Mean
- **Tests:** t-Test, Wilcoxon, Patell, BMP, Kolari-Pynnönen (Korrektur für Event-Clustering), Rangtest (Corrado/Campbell-Wasley), generalisierter Vorzeichentest
- **Robustheit:**
  - alternative Modelle und Ausreißerbehandlung (Winsorisieren/Trimmen)
  - Untergruppen, z. B. nach KI-Exposition
  - Differenztests gegen Kontrollgruppen
  - Placebo-Tests (Randomization Inference)
- **Methodik-Check** vor dem Start, z. B. für überlappende Fenster, ein Event im Schätzfenster oder kleine Gruppen
- **Protokoll** jedes Datenschritts: welches Unternehmen warum ausgeschlossen wurde, welches t=0 gilt usw.
- **Export:** Excel (alle Tabellen), Markdown-Bericht, Konfiguration als JSON; optional ein KI-Entwurf der Interpretation

## Installation & Start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Die App öffnet sich unter http://localhost:8501.

**KI-Hilfe (optional):** `.env.example` nach `.env` kopieren und `ANTHROPIC_API_KEY` eintragen. Ohne Key funktioniert
alles außer den KI-Vorschlägen.

## Daten

| Quelle | Wofür | Hinweis |
|---|---|---|
| Yahoo Finance | Kurse (Adjusted Close) und Indizes | Abruf in dieser Reihenfolge: `yfinance` → Yahoo-Chart-API direkt (wie R/quantmod, ohne Cookie/Crumb) → Stooq (nicht dividendenbereinigt, wird im Protokoll markiert); Ergebnisse landen in `cache/` |
| Kenneth R. French Data Library | FF3-Faktoren (Europe, US, Developed …) | Reihenfolge: Cache → ZIP in `data/factors/` (Browser-Download) → Website. Ohne Daten wird FF3 **übersprungen**, es werden keine Ersatzdaten erzeugt |
| Datenpaket `data/<name>/` | alle Kurse + Faktoren einer Studie | per `scripts/fetch_data.py` oder `scripts/fetch_data.R` erzeugt, siehe unten |
| CSV-Upload | eigene Kurse oder Renditen (Refinitiv, Bloomberg, CRSP …) | erste Spalte Datum, dann eine Spalte je Ticker inkl. Index |

### Wenn der Online-Abruf nicht klappt: Datenpaket

Das entspricht dem Weg „Daten in R laden, dann die Excel-Datei einlesen“, nur automatisiert:

```bash
# Studie im Assistenten als Entwurf speichern, dann auf einem Rechner mit Internet:
python scripts/fetch_data.py studies/<slug>/config.json          # oder eine Vorlage aus templates/
Rscript scripts/fetch_data.R studies/<slug>/config.json          # falls nur R an Yahoo kommt
```

Danach im Assistenten in Schritt 6 „Datenpaket“ wählen. Ein Paket in `data/` kann auch committet
werden; dann lässt sich die Studie z. B. in einer Cloud-Session ohne Internetzugang rechnen.

Unternehmensuniversen für die angeleitete Auswahl liegen in `universes/` (CSV mit mindestens einer `ticker`-Spalte).
`bachelorarbeit_stichprobe.csv` enthält die 408 Unternehmen der Arbeit mit Region, Sektor und KI-Exposition.

## Validierung gegen die Bachelorarbeit

```bash
python scripts/replicate_thesis.py /pfad/zu/EU-AI-Act-Event-Study/Data --save
```

Das Skript rechnet mit den Renditedateien aus dem Thesis-Repository und vergleicht CAAR und t-Statistik mit den
Tabellen 1 und 2 für beide Events. Alle 15 geprüften Werte stimmen exakt überein. Bei US-Tech wird mit der
Ergebnisdatei im Repository verglichen (N=185); die PDF-Werte (N=186) stammen aus einem früheren Datenabruf.
Mit `--save` erscheint die Replikation als Studie im Dashboard.

## Tests

```bash
pytest
```

Die Tests prüfen die Engine mit synthetischen Daten, in die ein bekannter Schock eingebaut ist.

## Projektstruktur

```
app.py                  Streamlit-Einstieg (Navigation)
views/                  Seiten: Dashboard, Assistent, Ergebnisse, Methodik
ui/                     Gruppen-Editor, Charts, Session-State
eventstudy/             Rechenkern (ohne UI nutzbar)
  config.py             StudyConfig – alles, was der Assistent erfasst (JSON)
  data.py               Download, Renditen, Bereinigung, FF3-Faktoren
  estimation.py         Normalrendite-Modelle, AR/CAR, standardisierte CARs
  significance.py       Signifikanztests, Ausreißer, Differenztests
  placebo.py            Placebo-Tests / Randomization Inference
  engine.py             Orchestrierung → StudyResult
  checks.py             Methodik-Check
  report.py             Tabellen, Excel, Markdown
  universe.py           Unternehmensuniversen und Filter
  ai.py                 optionale Claude-Vorschläge
templates/              Vorlagen (EU AI Act)
universes/              Unternehmenslisten für die angeleitete Auswahl
scripts/                Replikation der Bachelorarbeit
tests/                  pytest
```

## Ohne Oberfläche nutzen

```python
from eventstudy import StudyConfig, run_study
from eventstudy.report import results_table

cfg = StudyConfig.load("templates/eu_ai_act.json")
res = run_study(cfg)               # lädt Kurse von Yahoo Finance
print(results_table(res))
```
