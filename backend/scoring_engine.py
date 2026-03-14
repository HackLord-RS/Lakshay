"""
LAKSHAY — Psychometric Scoring Engine
=======================================
Converts raw questionnaire answers into a normalised 7-dimensional
personality / skill profile vector.

Dimensions
----------
  analytical  — logical reasoning, quantitative aptitude (RIASEC: I)
  technical   — engineering, coding, hands-on construction (RIASEC: R)
  creative    — artistic expression, design, novel ideation (RIASEC: A)
  social      — empathy, helping, interpersonal connection (RIASEC: S)
  leadership  — initiative, persuasion, team management   (RIASEC: E)
  business    — entrepreneurship, commerce, strategy      (RIASEC: E/C)
  research    — curiosity, investigation, academic depth  (RIASEC: I)

Algorithm
---------
1. Aggregate weighted raw scores per dimension.
   situational_judgment × 1.25 (reveals actual behaviour, not stated preference)
   personality_trait    × 1.15
   aptitude_check       × 1.10
   cognitive_style      × 1.05
   interest_mapping     × 1.00

2. Normalise against observed maximum to 0–100.

3. Apply confidence regression toward 45 (neutral) for sessions
   with fewer questions — prevents overconfident profiles from
   partial assessments.

4. Map dominant dimensions to RIASEC 3-letter code.

References
----------
  Holland (1973) — RIASEC model
  Schmidt & Hunter (1998) — Situational judgment test validity
  Kolb (1984) — Experiential learning styles
"""

from typing import Optional

DIMENSIONS = ["analytical", "technical", "creative", "social", "leadership", "business", "research"]

# ── RIASEC mapping ────────────────────────────────────────────────────
_DIM_TO_RIASEC = {
    "analytical": "I",
    "technical":  "R",
    "creative":   "A",
    "social":     "S",
    "leadership": "E",
    "business":   "E",
    "research":   "I",
}

# ── Profile-type labels ───────────────────────────────────────────────
_PROFILE_LABELS = {
    ("analytical", "technical"):  "Analytical–Technical",
    ("analytical", "research"):   "Analytical–Research",
    ("technical",  "analytical"): "Technical–Analytical",
    ("technical",  "creative"):   "Technical–Creative",
    ("creative",   "social"):     "Creative–Social",
    ("creative",   "analytical"): "Creative–Analytical",
    ("creative",   "technical"):  "Creative–Technical",
    ("social",     "leadership"): "Social–Leader",
    ("social",     "creative"):   "Social–Creative",
    ("social",     "research"):   "Social–Research",
    ("leadership", "business"):   "Leader–Entrepreneur",
    ("leadership", "social"):     "Social–Leader",
    ("business",   "leadership"): "Entrepreneurial",
    ("business",   "analytical"): "Business–Analytical",
    ("research",   "analytical"): "Research–Analytical",
    ("research",   "social"):     "Research–Social",
    ("research",   "technical"):  "Research–Technical",
}

_FULL_RIASEC_NAMES = {
    "R": "Realistic",
    "I": "Investigative",
    "A": "Artistic",
    "S": "Social",
    "E": "Enterprising",
    "C": "Conventional",
}


def calculate_profile(scored_answers: list) -> dict:
    """
    Build the normalised psychometric profile.

    Parameters
    ----------
    scored_answers : list
        Each element: {question_id, type, type_weight, scores: {dim: int}}

    Returns
    -------
    dict  {
        "normalized_scores":   {dim: int 0-100},
        "raw_scores":          {dim: float},
        "dominant_traits":     [str] × 3,
        "secondary_traits":    [str] × 2,
        "profile_type":        str,
        "riasec_code":         str  (3-letter),
        "riasec_full":         str  (expanded names),
        "confidence":          float 0-1,
        "questions_answered":  int,
    }
    """
    if not scored_answers:
        return _neutral_profile()

    # ── 1. Weighted aggregation ───────────────────────────────────────
    raw = {d: 0.0 for d in DIMENSIONS}
    for ans in scored_answers:
        w = ans.get("type_weight", 1.0)
        for dim in DIMENSIONS:
            raw[dim] += ans["scores"].get(dim, 0) * w

    n = len(scored_answers)

    # ── 2. Normalise to 0–100 ─────────────────────────────────────────
    max_raw = max(raw.values()) or 1.0
    normalised_raw = {d: (raw[d] / max_raw) * 95 for d in DIMENSIONS}

    # ── 3. Confidence regression ──────────────────────────────────────
    # Full confidence at 15 questions; at fewer, scores regress to 45 (neutral)
    cf = min(1.0, n / 15.0)
    normalised = {
        d: max(10, min(99, round(45 + (normalised_raw[d] - 45) * cf)))
        for d in DIMENSIONS
    }

    # ── 4. Rank & label ───────────────────────────────────────────────
    ranked = sorted(normalised.items(), key=lambda x: x[1], reverse=True)
    dominant  = [r[0] for r in ranked[:3]]
    secondary = [r[0] for r in ranked[3:5]]

    profile_type = _label_profile(dominant)
    riasec_code  = _to_riasec(dominant)
    riasec_full  = " + ".join(
        _FULL_RIASEC_NAMES.get(c, c) for c in riasec_code
    )

    return {
        "normalized_scores":  normalised,
        "raw_scores":         {d: round(raw[d], 1) for d in DIMENSIONS},
        "dominant_traits":    dominant,
        "secondary_traits":   secondary,
        "profile_type":       profile_type,
        "riasec_code":        riasec_code,
        "riasec_full":        riasec_full,
        "confidence":         round(cf, 2),
        "questions_answered": n,
    }


# ── Helpers ────────────────────────────────────────────────────────────

def _label_profile(dominant: list) -> str:
    if len(dominant) >= 2:
        key = (dominant[0], dominant[1])
        return _PROFILE_LABELS.get(
            key,
            f"{dominant[0].capitalize()}–{dominant[1].capitalize()}"
        )
    return dominant[0].capitalize() if dominant else "Balanced"


def _to_riasec(dominant: list) -> str:
    seen, code = set(), []
    for d in dominant:
        r = _DIM_TO_RIASEC.get(d, "C")
        if r not in seen:
            seen.add(r)
            code.append(r)
    return "".join(code[:3]) or "RIA"


def _neutral_profile() -> dict:
    return {
        "normalized_scores":  {d: 50 for d in DIMENSIONS},
        "raw_scores":         {d: 0.0 for d in DIMENSIONS},
        "dominant_traits":    ["analytical", "technical", "creative"],
        "secondary_traits":   ["social", "leadership"],
        "profile_type":       "Balanced",
        "riasec_code":        "RIA",
        "riasec_full":        "Realistic + Investigative + Artistic",
        "confidence":         0.0,
        "questions_answered": 0,
    }