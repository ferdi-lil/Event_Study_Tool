import streamlit as st

st.title("Methodik")
st.markdown(r"""
Das Tool folgt dem Event-Study-Design nach **MacKinlay (1997)**.

### 1. Normalrendite und abnormale Rendite
Für jedes Unternehmen *i* werden die Modellparameter im **Schätzfenster** (Standard: −250 bis −11 Handelstage) per OLS geschätzt.

| Modell | Erwartete Rendite |
|---|---|
| Marktmodell | $\hat r_{i,t} = \hat\alpha_i + \hat\beta_i r_{m,t}$ |
| Fama-French 3 | $\hat r_{i,t} - r_{f,t} = \hat\alpha_i + \hat\beta_{m}(r_{m,t}-r_{f,t}) + \hat\beta_{s} SMB_t + \hat\beta_{h} HML_t$ |
| Marktbereinigt | $\hat r_{i,t} = r_{m,t}$ |
| Constant Mean | $\hat r_{i,t} = \bar r_i$ |

$AR_{i,t} = r_{i,t} - \hat r_{i,t}$, $\;CAR_i(\tau_1,\tau_2)=\sum_{t=\tau_1}^{\tau_2} AR_{i,t}$, $\;CAAR = \frac{1}{N}\sum_i CAR_i$.

t=0 ist der erste Handelstag am oder nach dem Event-Datum. Alle Fenster zählen **Handelstage** im gemeinsamen Kalender von Gruppe und Marktindex.

### 2. Signifikanztests
| Test | Idee | Wann wichtig |
|---|---|---|
| t-Test | $t = CAAR / (s_{CAR}/\sqrt N)$ | Standard |
| Wilcoxon | Rangtest auf den Median der CARs | Ausreißer, Schiefe |
| Patell (1976) | Summe standardisierter CARs (inkl. Prognosefehler-Korrektur) | Unterschiedliche Volatilitäten |
| BMP (Boehmer et al. 1991) | t-Test auf standardisierte CARs | ereignisbedingte Volatilität |
| **Kolari & Pynnönen (2010)** | BMP × $\sqrt{(1-\bar r)/(1+(N-1)\bar r)}$, $\bar r$ = mittlere Residuenkorrelation | **gleiches Event-Datum für alle Firmen** |
| Rangtest (Corrado 1989; Campbell & Wasley 1993) | Ränge der ARs über Schätz- und Eventfenster | nicht-normale Renditen |
| Generalisierter Vorzeichentest (Cowan 1992) | Anteil positiver CARs vs. Anteil im Schätzfenster | robust, einfach |

### 3. Robustheit
- **Alternative Modelle** (z. B. FF3) – bleibt der Effekt modellunabhängig?
- **Ausreißer** – Winsorisieren oder Trimmen der CARs.
- **Kontrollgruppen** – Differenztest (Welch, Mann-Whitney) zwischen Untersuchungs- und Kontrollgruppe.
- **Placebo-Tests / Randomization Inference** (Brown & Warner 1980, 1985) – dieselbe Analyse an *N* zufälligen Tagen ohne Ereignis (mit Mindestabstand zu den echten Events). Perzentil und empirischer p-Wert zeigen, wie ungewöhnlich der echte Effekt ist.

### 4. Datenbereinigung (alles im Protokoll)
1. Ticker ohne Kursdaten werden ausgeschlossen (z. B. delistet).
2. Mindest-Datenabdeckung über den Gesamtzeitraum (Standard 80 %).
3. Mindestbeobachtungen im Schätzfenster.
4. Optional: vollständige Kurse im Eventfenster.
5. Optional: extreme Tagesrenditen (|log r| > 1) werden als Datenfehler entfernt.

### Grenzen
Kostenlose Kursdaten (Yahoo Finance) enthalten keine dekotierten Unternehmen (Survivorship Bias) und werden gelegentlich rückwirkend korrigiert. Für Publikationen sind lizenzierte Daten (Refinitiv, Bloomberg, CRSP) per CSV-Upload vorzuziehen.
""")
