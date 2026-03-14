"""
LAKSHAY — Career Knowledge Database
=====================================
Loads careers.json and provides structured query functions.
Enriches ML prediction results with full career metadata.
"""

import json
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
_cache: Optional[dict] = None


def _db() -> dict:
    global _cache
    if _cache is None:
        with open(DATA_DIR / "careers.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        _cache = {c["id"]: c for c in data["careers"]}
    return _cache


def get_career(career_id: str) -> Optional[dict]:
    return _db().get(career_id)


def all_careers() -> list:
    return list(_db().values())


def careers_by_stream(stream_fragment: str) -> list:
    key = stream_fragment.lower()
    return [c for c in _db().values() if key in c.get("stream", "").lower()]


def enrich(predictions: list) -> list:
    """
    Merge ML prediction dicts with full career metadata.

    Parameters
    ----------
    predictions : list  Output of recommendation_model.predict()

    Returns
    -------
    list  Enriched dicts ready for the Result API.
    """
    db = _db()
    enriched = []

    for p in predictions:
        cid  = p["career_id"]
        data = db.get(cid, {})

        enriched.append({
            # ML prediction fields
            "rank":        p["rank"],
            "career_id":   cid,
            "career_name": data.get("name", p["career_name"]),
            "emoji":       data.get("emoji", ""),
            "match_score": p["match_score"],
            "confidence":  p["confidence"],

            # Knowledge-base fields
            "category":           data.get("category", ""),
            "stream":             data.get("stream", ""),
            "description":        data.get("description", ""),
            "required_skills":    data.get("required_skills", []),
            "soft_skills":        data.get("soft_skills", []),
            "education_path":     data.get("education_path", []),
            "top_colleges_india": data.get("top_colleges_india", []),
            "entrance_exams":     data.get("entrance_exams", []),
            "avg_salary_inr":     data.get("avg_salary_inr", {}),
            "avg_salary_global_usd": data.get("avg_salary_global_usd", {}),
            "future_demand":      data.get("future_demand", "Moderate"),
            "demand_growth_pct":  data.get("demand_growth_pct", 20),
            "global_scope":       data.get("global_scope", True),
            "job_market_note":    data.get("job_market_note", ""),
            "riasec":             data.get("riasec", []),
            "suggested_next_skills": data.get("suggested_next_skills", []),
            "work_environment":   data.get("work_environment", ""),
            "daily_tasks":        data.get("daily_tasks", []),
        })

    return enriched