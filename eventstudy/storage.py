"""Studies live in studies/<slug>/: config.json, result.pkl and exported files."""
from __future__ import annotations

import pickle
import shutil
from datetime import datetime
from pathlib import Path

from .config import StudyConfig

ROOT = Path(__file__).resolve().parent.parent
STUDIES_DIR = ROOT / "studies"
TEMPLATES_DIR = ROOT / "templates"


def study_dir(slug: str) -> Path:
    return STUDIES_DIR / slug


def unique_slug(cfg: StudyConfig) -> str:
    base, slug, i = cfg.slug, cfg.slug, 2
    while (STUDIES_DIR / slug).exists():
        slug, i = f"{base}-{i}", i + 1
    return slug


def save_config(cfg: StudyConfig, slug: str | None = None) -> str:
    slug = slug or unique_slug(cfg)
    d = study_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    cfg.save(d / "config.json")
    return slug


def save_result(slug: str, result) -> None:
    with open(study_dir(slug) / "result.pkl", "wb") as f:
        pickle.dump(result, f)


def load_config(slug: str) -> StudyConfig:
    return StudyConfig.load(study_dir(slug) / "config.json")


def load_result(slug: str):
    p = study_dir(slug) / "result.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def delete_study(slug: str) -> None:
    shutil.rmtree(study_dir(slug), ignore_errors=True)


def list_studies() -> list[dict]:
    if not STUDIES_DIR.exists():
        return []
    out = []
    for d in STUDIES_DIR.iterdir():
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = StudyConfig.load(cfg_path)
        except Exception:
            continue
        res = d / "result.pkl"
        out.append({
            "slug": d.name, "config": cfg,
            "status": "Ausgewertet" if res.exists() else "Entwurf",
            "updated": datetime.fromtimestamp(max(cfg_path.stat().st_mtime,
                                                  res.stat().st_mtime if res.exists() else 0)),
        })
    return sorted(out, key=lambda s: s["updated"], reverse=True)


def list_templates() -> list[tuple[str, StudyConfig]]:
    return [(p.stem, StudyConfig.load(p)) for p in sorted(TEMPLATES_DIR.glob("*.json"))]
