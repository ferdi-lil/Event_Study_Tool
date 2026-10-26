"""Finder: from a research direction to concrete, checkable study ideas.

Stage 1 of the funnel. Ideas are built from metadata only (event catalog or Claude); no
prices are loaded. Groups are resolved via the screener only when an idea is pursued.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from .config import Event, Group, StudyConfig
from .screener import COUNTRIES, REGIONS, SECTORS, UniverseQuery, expand_regions, industries_by_sector, \
    suggest_index_for

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog" / "events.json"

EVENT_TYPES = ["Regulierung", "Handel & Zölle", "Geopolitik", "Wahlen & Politik", "Geld- & Fiskalpolitik",
               "Finanzsystem", "Technologie", "Unternehmen & Skandale", "Gesundheit & Pandemie",
               "Energie & Rohstoffe"]
PURPOSES = ["Stellungnahme / Positionspapier", "Mitgliederinformation", "Presse / Story", "Forschung"]
EUROPE = set(REGIONS["Europa (EU + UK, CH, NO)"])
SURPRISE_SCORE = {"hoch": 2, "mittel": 1, "gering": 0}


@dataclass
class Direction:
    theme: str = ""
    sectors: list[str] = field(default_factory=list)  # Yahoo sector names
    industries: list[str] = field(default_factory=list)  # Yahoo industry names
    regions: list[str] = field(default_factory=list)  # region or country labels
    start: str = "2015-01-01"
    end: str = field(default_factory=lambda: date.today().isoformat())
    event_types: list[str] = field(default_factory=list)
    purpose: str = PURPOSES[0]

    def countries(self) -> list[str]:
        return expand_regions(self.regions)

    def sector_set(self) -> set[str]:
        """Sectors including those implied by chosen industries."""
        by_ind = {i: s for s, inds in industries_by_sector().items() for i in inds}
        return set(self.sectors) | {by_ind[i] for i in self.industries if i in by_ind}

    def label(self) -> str:
        what = ", ".join(self.industries) or ", ".join(SECTORS.get(s, s) for s in self.sectors) or "alle Sektoren"
        return f"{what} · {', '.join(self.regions) or 'alle Regionen'}"


@dataclass
class GroupSpec:
    name: str
    role: str  # treatment | control
    query: UniverseQuery
    rationale: str = ""


@dataclass
class Idea:
    id: str
    title: str
    question: str
    events: list[dict]  # {date, label, source, confidence}
    event_type: str
    mechanism: str
    expected_direction: str
    surprise: str
    groups: list[GroupSpec]
    origin: str  # "Katalog" | "KI"
    score: float = 0.0
    confounders: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ catalog
def load_catalog(path: Path = CATALOG) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["events"]


def _regions_match(event_regions: list[str], countries: list[str]) -> bool:
    if not countries or "Global" in event_regions:
        return True
    return bool(set(expand_regions(event_regions)) & set(countries))


def _short(s: str) -> str:
    return SECTORS.get(s, s)


def default_groups(d: Direction) -> list[GroupSpec]:
    """Treatment = the direction itself; controls = same sector elsewhere, other sectors here."""
    countries = d.countries() or REGIONS["EU"]
    region_label = ", ".join(d.regions) or "EU"
    what = ", ".join(d.industries) or ", ".join(_short(s) for s in d.sectors) or "Alle Sektoren"
    groups = [GroupSpec(f"{what} · {region_label}", "treatment",
                        UniverseQuery(countries, sectors=list(d.sectors), industries=list(d.industries)),
                        "Unternehmen der gewählten Richtung")]
    if d.sectors or d.industries:
        if set(countries) <= EUROPE:
            other, other_label = ["USA"], "USA"
        else:
            other, other_label = REGIONS["EU"], "EU"
        groups.append(GroupSpec(f"{what} · {other_label}", "control",
                                UniverseQuery(other, sectors=list(d.sectors), industries=list(d.industries)),
                                "Gleiche Branche, andere Region: schließt einen globalen Branchenschock aus"))
        groups.append(GroupSpec(f"Übrige Sektoren · {region_label}", "control",
                                UniverseQuery(countries, exclude_sectors=sorted(d.sector_set()), max_results=60),
                                "Gleiche Region, andere Branchen (60 größte): schließt eine allgemeine "
                                "Marktbewegung aus"))
    return groups


def catalog_ideas(d: Direction, catalog: list[dict] | None = None, limit: int = 12) -> list[Idea]:
    catalog = catalog if catalog is not None else load_catalog()
    countries = d.countries()
    want_sectors = d.sector_set()
    ideas = []
    for ev in catalog:
        if not (d.start <= ev["date"] <= d.end):
            continue
        if d.event_types and ev["type"] not in d.event_types:
            continue
        if not _regions_match(ev["regions"], countries):
            continue
        ev_sectors = set(ev.get("sectors", []))
        ev_inds = set(ev.get("industries", []))
        if want_sectors and "*" not in ev_sectors and not (ev_sectors & want_sectors):
            continue
        score = SURPRISE_SCORE.get(ev.get("surprise"), 0)
        score += 3 if d.industries and ev_inds & set(d.industries) else 0
        score += 2 if want_sectors and ev_sectors & want_sectors else 0.5 if "*" in ev_sectors else 0
        score += 1 if countries and set(expand_regions(ev["regions"])) & set(countries) else 0
        score += 1 if d.event_types and ev["type"] in d.event_types else 0
        groups = default_groups(d)
        treat = groups[0].name
        ideas.append(Idea(
            id=f"kat-{ev['date']}", title=f"{ev['label']} – Reaktion: {treat}",
            question=(f"Haben Anleger {treat} rund um „{ev['label']}“ anders bewertet als "
                      f"{' bzw. '.join(g.name for g in groups[1:]) or 'den Gesamtmarkt'}?"),
            events=[{"date": ev["date"], "label": ev["label"], "source": "Kuratierter Katalog",
                     "confidence": "Katalog – bitte prüfen"}],
            event_type=ev["type"], mechanism=ev.get("description", ""), expected_direction="offen",
            surprise=ev.get("surprise", "mittel"), groups=groups, origin="Katalog", score=score))
    ideas.sort(key=lambda i: (-i.score, i.events[0]["date"]))
    ideas = ideas[:limit]
    for idea in ideas:
        idea.confounders = confounders(idea, catalog)
    return ideas


def confounders(idea: Idea, catalog: list[dict], days: int = 7) -> list[str]:
    """Other catalogued events within ±days of the idea's events."""
    own = {e["date"] for e in idea.events}
    out = []
    for e in idea.events:
        t = pd.Timestamp(e["date"])
        for c in catalog:
            if c["date"] not in own and abs((pd.Timestamp(c["date"]) - t).days) <= days:
                out.append(f"{c['date']}: {c['label']}")
    return sorted(set(out))


# ------------------------------------------------------------------- AI
def ai_ideas(d: Direction, catalog: list[dict] | None = None) -> list[Idea]:
    from . import ai

    raw = ai.suggest_ideas(d.theme, d.label(), d.start, d.end, d.event_types, d.purpose,
                           sorted(SECTORS), sorted({i for v in industries_by_sector().values() for i in v}),
                           sorted(COUNTRIES) + sorted(REGIONS))
    catalog = catalog if catalog is not None else load_catalog()
    valid_s, valid_i = set(SECTORS), {i for v in industries_by_sector().values() for i in v}
    ideas = []
    for k, r in enumerate(raw.get("ideas", [])):
        groups = []
        for g in r.get("groups", []):
            regs = [x for x in g.get("regions", []) if x in COUNTRIES or x in REGIONS] or d.regions or ["EU"]
            q = UniverseQuery(regs, sectors=[s for s in g.get("sectors", []) if s in valid_s],
                              industries=[i for i in g.get("industries", []) if i in valid_i],
                              exclude_sectors=[s for s in g.get("exclude_sectors", []) if s in valid_s],
                              max_results=250 if g.get("role") == "treatment" else 80)
            groups.append(GroupSpec(g.get("name") or f"Gruppe {len(groups) + 1}", g.get("role", "control"), q,
                                    g.get("rationale", "")))
        if not any(g.role == "treatment" for g in groups):
            groups = default_groups(d)
        events = [e for e in r.get("events", []) if _valid_date(e.get("date"))]
        if not events:
            continue
        idea = Idea(id=f"ki-{k}-{events[0]['date']}", title=r.get("title", ""), question=r.get("question", ""),
                    events=events, event_type=r.get("event_type", ""), mechanism=r.get("mechanism", ""),
                    expected_direction=r.get("expected_direction", "offen"), surprise=r.get("surprise", "mittel"),
                    groups=groups, origin="KI", caveats=r.get("caveats", []))
        idea.confounders = confounders(idea, catalog)
        if any(e.get("confidence") != "sicher" for e in events):
            idea.caveats.insert(0, "Mindestens ein Datum ist von der KI als unsicher markiert.")
        ideas.append(idea)
    return ideas


def _valid_date(s) -> bool:
    try:
        return pd.Timestamp(s) <= pd.Timestamp.today()
    except (ValueError, TypeError):
        return False


# ------------------------------------------------------------ to a study
def idea_to_config(idea: Idea, universes: dict[str, pd.DataFrame], direction: Direction | None = None) -> StudyConfig:
    """Build a ready-to-run study. Groups whose universe is empty are left out (and reported)."""
    groups, missing = [], []
    for spec in idea.groups:
        df = universes.get(spec.name)
        if df is None or df.empty:
            missing.append(spec.name)
            continue
        countries = expand_regions(spec.query.countries)
        idx, ff3 = suggest_index_for(sorted(set(df["country"].dropna())) or countries)
        many_industries = spec.role == "treatment" and df["industry"].nunique() > 1
        groups.append(Group(
            name=spec.name, tickers=df["ticker"].tolist(), role=spec.role, market_index=idx, ff3_region=ff3,
            names=dict(zip(df["ticker"], df["name"].fillna(""))),
            subgroups=dict(zip(df["ticker"], df["industry"].fillna(""))) if many_industries else {},
            selection_note=f"{spec.rationale}. {spec.query.describe()}".strip(". "),
            meta={r.ticker: {"country": r.country, "industry": r.industry, "market_cap": r.market_cap,
                             "currency": r.currency} for r in df.itertuples()}))
    desc = f"{idea.question}\n\nEntstanden im Finder ({idea.origin}-Idee)."
    if direction:
        desc += f" Richtung: {direction.label()}; Zweck: {direction.purpose}."
    if missing:
        desc += f" Ohne Unternehmen und daher nicht angelegt: {', '.join(missing)}."
    cfg = StudyConfig(name=idea.title[:90], description=desc,
                      events=[Event(e["date"], e.get("label", "")) for e in idea.events], groups=groups)
    cfg.placebo.groups = [g.name for g in groups if g.role == "treatment"]
    return cfg
