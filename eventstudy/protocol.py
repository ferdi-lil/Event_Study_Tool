"""Methodology protocol: every data decision is recorded so the study is reproducible."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Entry:
    step: str
    message: str
    level: str = "info"  # info | warning | exclusion
    details: dict = field(default_factory=dict)
    time: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class Protocol:
    def __init__(self) -> None:
        self.entries: list[Entry] = []

    def info(self, step: str, message: str, **details) -> None:
        self.entries.append(Entry(step, message, "info", details))

    def warn(self, step: str, message: str, **details) -> None:
        self.entries.append(Entry(step, message, "warning", details))

    def exclude(self, step: str, ticker: str, reason: str, **details) -> None:
        self.entries.append(Entry(step, f"{ticker} ausgeschlossen: {reason}", "exclusion", {"ticker": ticker, **details}))

    def exclusions(self) -> list[Entry]:
        return [e for e in self.entries if e.level == "exclusion"]

    def warnings(self) -> list[Entry]:
        return [e for e in self.entries if e.level == "warning"]

    def to_records(self) -> list[dict]:
        return [
            {"Zeit": e.time, "Schritt": e.step, "Typ": e.level, "Meldung": e.message}
            for e in self.entries
        ]

    def to_markdown(self) -> str:
        lines = []
        current = None
        for e in self.entries:
            if e.step != current:
                lines.append(f"\n### {e.step}\n")
                current = e.step
            prefix = {"warning": "- **Warnung:** ", "exclusion": "- **Ausschluss:** "}.get(e.level, "- ")
            lines.append(f"{prefix}{e.message}")
        return "\n".join(lines).strip()
