"""Optional Claude assistant: suggestions for events, groups, controls and a results draft.

Everything here is a *suggestion* the user confirms in the UI; nothing is applied silently.
Requires the `anthropic` package and an API key (ANTHROPIC_API_KEY, e.g. via .env).
"""
from __future__ import annotations

import json
import os

MODEL = os.environ.get("EVENTSTUDY_CLAUDE_MODEL", "claude-opus-5-5")

SYSTEM = (
    "Du bist ein empirischer Kapitalmarktforscher und unterstützt bei Event Studies nach MacKinlay (1997). "
    "Antworte auf Deutsch, präzise und methodisch sauber. Erfinde keine Fakten: wenn du dir bei einem Datum "
    "oder Ticker nicht sicher bist, sag das im Feld für Hinweise. Ticker im Yahoo-Finance-Format "
    "(z. B. SAP.DE, ASML.AS, MSFT)."
)


class AIUnavailable(RuntimeError):
    pass


def available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _client():
    try:
        import anthropic
    except ImportError as exc:
        raise AIUnavailable("Paket 'anthropic' ist nicht installiert") from exc
    if not available():
        raise AIUnavailable("Kein ANTHROPIC_API_KEY gesetzt (siehe README)")
    return anthropic.Anthropic()


def _ask_json(prompt: str, schema: dict, effort: str = "high") -> dict:
    import anthropic

    client = _client()
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as exc:
        raise AIUnavailable("API-Key ungültig") from exc
    except anthropic.RateLimitError as exc:
        raise AIUnavailable("Rate-Limit erreicht – bitte kurz warten") from exc
    except anthropic.APIConnectionError as exc:
        raise AIUnavailable("Keine Verbindung zur Claude API") from exc
    except anthropic.APIStatusError as exc:
        raise AIUnavailable(f"API-Fehler {exc.status_code}: {exc.message}") from exc
    if response.stop_reason == "refusal":
        raise AIUnavailable("Die Anfrage wurde vom Modell abgelehnt")
    if response.stop_reason == "max_tokens":
        raise AIUnavailable("Antwort wurde abgeschnitten – Beschreibung bitte eingrenzen")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}


def suggest_events(topic: str) -> dict:
    schema = _obj({
        "events": {"type": "array", "items": _obj({
            "date": {"type": "string", "description": "YYYY-MM-DD"},
            "label": _STR, "why_relevant": _STR,
            "surprise": {"type": "string", "enum": ["hoch", "mittel", "gering"]},
            "confidence": {"type": "string", "enum": ["sicher", "unsicher"]},
        })},
        "recommendation": _STR,
        "caveats": _STRS,
    })
    prompt = (
        f"Thema der Event Study: {topic}\n\n"
        "Schlage die kapitalmarktrelevanten Schlüsselereignisse vor (max. 6). Bewerte jeweils den "
        "Überraschungsgehalt (Binder 1985: antizipierte Schritte haben wenig Informationsgehalt), nenne "
        "konfundierende Ereignisse in der Nähe und empfiehl, welche 1–3 Daten als Events dienen sollten. "
        "Ereignisse am Wochenende/nach Börsenschluss: Datum der Bekanntgabe angeben."
    )
    return _ask_json(prompt, schema)


def suggest_group(description: str, role: str, universe_info: str = "", context: str = "") -> dict:
    schema = _obj({
        "selection_criteria": _STR,
        "filters": {"type": "array", "items": _obj({"column": _STR, "values": _STRS})},
        "tickers": {"type": "array", "items": _obj({"ticker": _STR, "name": _STR, "reason": _STR})},
        "market_index": {"type": "string", "description": "Yahoo-Ticker des passenden Marktindex"},
        "rationale": _STR,
        "caveats": _STRS,
    })
    kind = "Untersuchungsgruppe" if role == "treatment" else "Kontrollgruppe"
    prompt = (
        f"Gesucht ist eine {kind} für eine Event Study.\nBeschreibung: {description}\n"
        + (f"Kontext (Events / Untersuchungsgruppe): {context}\n" if context else "")
        + (f"\nVerfügbares Unternehmensuniversum (Spalten und Werte):\n{universe_info}\n"
           "Nutze bevorzugt 'filters' mit exakt diesen Spaltennamen und Werten, damit die Auswahl "
           "reproduzierbar aus dem Universum entsteht. 'tickers' nur ergänzend.\n"
           if universe_info else
           "\nEs gibt kein Universum. Schlage konkrete börsennotierte Unternehmen als 'tickers' vor "
           "(10–40, nur solche, deren Yahoo-Ticker du sicher kennst), 'filters' leer lassen.\n")
        + "Formuliere die Auswahlkriterien so, dass sie in einer Abschlussarbeit zitierfähig sind."
    )
    return _ask_json(prompt, schema)


def suggest_controls(study_summary: str, universe_info: str = "") -> dict:
    schema = _obj({
        "control_groups": {"type": "array", "items": _obj({
            "name": _STR, "logic": _STR, "what_it_rules_out": _STR,
            "filters": {"type": "array", "items": _obj({"column": _STR, "values": _STRS})},
            "market_index": _STR,
        })},
        "caveats": _STRS,
    })
    prompt = (
        f"Studie: {study_summary}\n"
        + (f"Universum (Spalten und Werte):\n{universe_info}\n" if universe_info else "")
        + "Schlage 2–4 Kontrollgruppen vor, die alternative Erklärungen ausschließen (z. B. gleiche Region/andere "
          "Branche gegen allgemeine Marktbewegungen, gleiche Branche/andere Region gegen globale Sektorschocks). "
          "Erkläre jeweils, welche Erklärung die Gruppe ausschließt."
    )
    return _ask_json(prompt, schema)


def interpret_results(study_summary: str, results_text: str) -> str:
    schema = _obj({"interpretation_markdown": _STR})
    prompt = (
        f"Studie: {study_summary}\n\nErgebnisse:\n{results_text}\n\n"
        "Schreibe eine nüchterne Interpretation (ca. 300–500 Wörter, Markdown) für einen Ergebnisteil: "
        "Hauptbefund, Vergleich mit Kontrollgruppen, Robustheit (Modelle, Tests mit Clustering-Korrektur, "
        "Placebo), Grenzen. Keine Kausalbehauptungen über die Daten hinaus."
    )
    return _ask_json(prompt, schema, effort="medium")["interpretation_markdown"]


def suggest_ideas(theme: str, direction: str, start: str, end: str, event_types: list[str], purpose: str,
                  sectors: list[str], industries: list[str], regions: list[str]) -> dict:
    """Study ideas for the Finder. Sector/industry/region values must come from the given lists."""
    group = _obj({
        "name": _STR, "role": {"type": "string", "enum": ["treatment", "control"]},
        "regions": _STRS, "sectors": _STRS, "industries": _STRS, "exclude_sectors": _STRS, "rationale": _STR,
    })
    schema = _obj({"ideas": {"type": "array", "items": _obj({
        "title": _STR, "question": _STR,
        "events": {"type": "array", "items": _obj({
            "date": {"type": "string", "description": "YYYY-MM-DD, erster börsenrelevanter Tag"},
            "label": _STR, "source": {"type": "string", "description": "Wo sich das Datum belegen lässt"},
            "confidence": {"type": "string", "enum": ["sicher", "unsicher"]},
        })},
        "event_type": _STR, "mechanism": _STR,
        "expected_direction": {"type": "string", "enum": ["positiv", "negativ", "offen"]},
        "surprise": {"type": "string", "enum": ["hoch", "mittel", "gering"]},
        "groups": {"type": "array", "items": group},
        "caveats": _STRS,
    })}})
    prompt = (
        f"Richtung: {direction}\nThema/Interesse: {theme or '–'}\nZeitraum: {start} bis {end}\n"
        f"Ereignisarten: {', '.join(event_types) or 'alle'}\nZweck: {purpose}\n\n"
        "Schlage 5–8 konkrete Event-Study-Ideen vor, die mit Aktienkursen börsennotierter Unternehmen "
        "untersuchbar sind. Je Idee: ein klar datierbares Ereignis mit Überraschungsgehalt (Binder 1985: "
        "erwartete Schritte haben wenig Informationsgehalt), der vermutete Wirkungskanal, eine "
        "Untersuchungsgruppe und 1–2 Kontrollgruppen, die alternative Erklärungen ausschließen.\n"
        f"Erlaubte Regionen: {', '.join(regions)}\nErlaubte Sektoren: {', '.join(sectors)}\n"
        f"Erlaubte Branchen: {', '.join(industries)}\n"
        "Verwende ausschließlich Werte aus diesen Listen. Markiere ein Datum nur als 'sicher', wenn du es "
        "sicher weißt; bei Bekanntgabe nach Börsenschluss oder am Wochenende den Folgetag angeben."
    )
    return _ask_json(prompt, schema)
