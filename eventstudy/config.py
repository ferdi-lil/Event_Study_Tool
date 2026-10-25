"""Study configuration: everything the wizard collects, serialisable to JSON."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

RETURN_MODELS = {
    "market": "Marktmodell (OLS gegen Marktindex)",
    "market_adjusted": "Marktbereinigtes Modell (R_i - R_m)",
    "mean_adjusted": "Constant-Mean-Modell",
    "ff3": "Fama-French-3-Faktoren-Modell",
}

TESTS = {
    "t_test": "Querschnitts-t-Test",
    "wilcoxon": "Wilcoxon-Signed-Rank-Test",
    "patell": "Patell-Z (standardisierte ARs)",
    "bmp": "BMP-Test (Boehmer et al. 1991)",
    "kolari": "Kolari-Pynnönen-adjustierter BMP",
    "rank": "Rangtest (Corrado / Campbell-Wasley)",
    "sign": "Generalisierter Vorzeichentest (Cowan 1992)",
}

# Index suggestions per region; the user can always override them.
MARKET_INDICES = {
    "STOXX Europe 600": "^STOXX",
    "Euro Stoxx 50": "^STOXX50E",
    "DAX": "^GDAXI",
    "CAC 40": "^FCHI",
    "FTSE 100": "^FTSE",
    "S&P 500": "^GSPC",
    "Nasdaq Composite": "^IXIC",
    "Nikkei 225": "^N225",
    "MSCI World (ETF URTH)": "URTH",
}

FF3_REGIONS = {
    "Europe": "Europe_3_Factors_Daily_CSV.zip",
    "US": "F-F_Research_Data_Factors_daily_CSV.zip",
    "North_America": "North_America_3_Factors_Daily_CSV.zip",
    "Developed": "Developed_3_Factors_Daily_CSV.zip",
    "Japan": "Japan_3_Factors_Daily_CSV.zip",
    "Asia_Pacific_ex_Japan": "Asia_Pacific_ex_Japan_3_Factors_Daily_CSV.zip",
}


@dataclass
class Event:
    date: str  # ISO yyyy-mm-dd
    label: str = ""

    @property
    def timestamp(self):
        import pandas as pd

        return pd.Timestamp(self.date)


@dataclass
class Group:
    name: str
    tickers: list[str]
    role: str = "treatment"  # "treatment" or "control"
    market_index: str = "^STOXX"
    ff3_region: str = "Europe"
    names: dict[str, str] = field(default_factory=dict)
    # Optional subgroup label per ticker (e.g. AI exposure High/Middle/Low)
    subgroups: dict[str, str] = field(default_factory=dict)
    selection_note: str = ""  # how the group was built, goes into the protocol


@dataclass
class OutlierSettings:
    method: str = "none"  # none | winsorize | trim
    level: float = 0.01  # share per tail


@dataclass
class PlaceboSettings:
    enabled: bool = True
    n: int = 100
    seed: int = 39
    buffer_days: int = 60  # trading days kept clear around real events
    groups: list[str] = field(default_factory=list)  # empty = all treatment groups


@dataclass
class StudyConfig:
    name: str
    description: str = ""
    events: list[Event] = field(default_factory=list)
    groups: list[Group] = field(default_factory=list)
    windows: list[tuple[int, int]] = field(default_factory=lambda: [(-1, 1), (0, 1), (0, 3), (0, 5), (-5, 5)])
    estimation_window: tuple[int, int] = (-250, -11)
    model: str = "market"
    robustness_models: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=lambda: ["t_test", "wilcoxon", "bmp", "kolari", "rank"])
    return_type: str = "log"  # log | simple
    min_coverage: float = 0.8  # share of non-missing returns over the whole download period
    min_estimation_obs: int = 30
    require_full_event_window: bool = True
    # |log return| above this on a single day is treated as a data error (None = keep everything)
    extreme_return_threshold: float | None = 1.0
    outliers: OutlierSettings = field(default_factory=OutlierSettings)
    outlier_robustness: bool = True  # also report a winsorised variant next to the main spec
    placebo: PlaceboSettings = field(default_factory=PlaceboSettings)
    data_source: str = "yahoo"  # yahoo | upload
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # ----------------------------------------------------------------- helpers
    @property
    def slug(self) -> str:
        s = re.sub(r"[^a-zA-Z0-9]+", "-", self.name.lower()).strip("-")
        return s or "studie"

    def treatment_groups(self) -> list[Group]:
        return [g for g in self.groups if g.role == "treatment"]

    def control_groups(self) -> list[Group]:
        return [g for g in self.groups if g.role == "control"]

    def all_models(self) -> list[str]:
        return [self.model] + [m for m in self.robustness_models if m != self.model]

    def data_period(self) -> tuple[date, date]:
        """Calendar range that covers estimation + event windows of every event."""
        import pandas as pd

        dates = [e.timestamp for e in self.events]
        est_start = self.estimation_window[0]
        max_end = max(w[1] for w in self.windows)
        lead = abs(est_start) * 1.6 + 30  # trading -> calendar days, with margin
        if self.placebo.enabled:
            lead += 250 * 1.6
        start = min(dates) - pd.Timedelta(days=int(lead))
        end = max(dates) + pd.Timedelta(days=int(max_end * 1.6 + 30))
        if self.placebo.enabled:
            end += pd.Timedelta(days=200)
        return start.date(), min(end, pd.Timestamp.today().normalize()).date()

    @staticmethod
    def window_label(w: tuple[int, int]) -> str:
        return f"({w[0]:+d},{w[1]:+d})".replace("+0", "0")

    # ----------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        d = asdict(self)
        d["windows"] = [list(w) for w in self.windows]
        d["estimation_window"] = list(self.estimation_window)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StudyConfig":
        d = dict(d)
        d["events"] = [Event(**e) for e in d.get("events", [])]
        d["groups"] = [Group(**g) for g in d.get("groups", [])]
        d["windows"] = [tuple(w) for w in d.get("windows", [])]
        d["estimation_window"] = tuple(d.get("estimation_window", (-250, -11)))
        d["outliers"] = OutlierSettings(**d.get("outliers", {}))
        d["placebo"] = PlaceboSettings(**d.get("placebo", {}))
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StudyConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
