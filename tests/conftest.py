import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eventstudy.config import Event, Group, StudyConfig  # noqa: E402

EVENT = "2024-03-13"


@pytest.fixture
def synthetic():
    """Prices for 40 treated and 30 control firms; treated lose 3 % on t=0."""
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2022-06-01", "2024-12-31")
    mkt = rng.normal(0.0003, 0.01, len(days))
    rets = {"^STOXX": mkt}
    e = days.searchsorted(pd.Timestamp(EVENT))
    for prefix, n, shock in (("T", 40, -0.03), ("C", 30, 0.0)):
        for i in range(n):
            beta = rng.uniform(0.6, 1.4)
            r = 0.0001 + beta * mkt + rng.normal(0, 0.012, len(days))
            r[e] += shock
            rets[f"{prefix}{i}.DE"] = r
    returns = pd.DataFrame(rets, index=days)
    prices = 100 * np.exp(returns.cumsum())
    prices.iloc[:400, prices.columns.get_loc("T39.DE")] = np.nan  # listed too late -> excluded
    factors = pd.DataFrame({"Mkt-RF": mkt, "SMB": rng.normal(0, 0.004, len(days)),
                            "HML": rng.normal(0, 0.004, len(days)), "RF": 0.00005}, index=days)
    return prices, factors


@pytest.fixture
def config():
    return StudyConfig(
        name="Synthetisch",
        events=[Event(EVENT, "Test")],
        groups=[Group("Treated", [f"T{i}.DE" for i in range(40)], "treatment", "^STOXX",
                      subgroups={f"T{i}.DE": "A" if i < 20 else "B" for i in range(40)}),
                Group("Control", [f"C{i}.DE" for i in range(30)], "control", "^STOXX")],
        windows=[(0, 0), (-1, 1), (0, 5)],
        tests=["t_test", "wilcoxon", "patell", "bmp", "kolari", "rank", "sign"],
    )
