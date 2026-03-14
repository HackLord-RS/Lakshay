"""
LAKSHAY — Full Analysis Orchestrator
======================================
Chains all modules into a single  run_analysis()  call:

  Scored answers
       ↓
  Psychometric profile  (scoring_engine)
       ↓
  ML career predictions (recommendation_model)
       ↓
  Enriched career data  (career_db)
       ↓
  Natural-language explanations (explanation_engine)
       ↓
  Complete dashboard payload  ← returned to API
"""

from .questionnaire_engine  import get_scored_answers
from .scoring_engine         import calculate_profile
from .recommendation_model   import predict, get_stream, train
from .career_db              import enrich
from .explanation_engine     import (
    explain, stream_rationale, skill_suggestions
)


def run_analysis(session_id: str) -> dict:
    """
    Execute the complete LAKSHAY AI pipeline for a session.

    Returns
    -------
    dict  Complete dashboard payload for the frontend.
          Contains: recommended_stream, top_careers (enriched + explained),
          skill_profile, market_insights, suggested_skills, summary.
    """

    # ── 1. Retrieve scored answers ──────────────────────────────────
    scored_data = get_scored_answers(session_id)
    if not scored_data:
        return {"error": "Session not found or expired.", "session_id": session_id}

    if not scored_data["scored_answers"]:
        return {"error": "No answers recorded for this session.", "session_id": session_id}

    # ── 2. Psychometric scoring ─────────────────────────────────────
    profile = calculate_profile(scored_data["scored_answers"])

    # ── 3. ML prediction ────────────────────────────────────────────
    raw_preds = predict(profile["normalized_scores"], top_n=5)

    # ── 4. Enrich with career knowledge base ───────────────────────
    careers = enrich(raw_preds)

    # ── 5. Determine recommended stream ─────────────────────────────
    top_id = careers[0]["career_id"] if careers else "data_scientist"
    stream_name, edu_steps = get_stream(top_id)

    # ── 6. Explanations ─────────────────────────────────────────────
    for c in careers:
        c["explanation"] = explain(
            career_id=c["career_id"],
            career_name=c["career_name"],
            profile=profile,
            rank=c["rank"],
        )

    stream_rat  = stream_rationale(stream_name, profile)
    skills_list = skill_suggestions(careers, profile)

    # ── 7. Market insights ──────────────────────────────────────────
    top = careers[0] if careers else {}

    # ── 8. Assemble final payload ───────────────────────────────────
    return {
        "session_id": session_id,
        "user_type":  scored_data["user_type"],
        "user_role":  scored_data["user_role"],

        # ── Stream recommendation ──
        "recommended_stream":  stream_name,
        "stream_rationale":    stream_rat,
        "stream_education_path": edu_steps,

        # ── Top careers ──
        "top_careers": [
            {
                "rank":            c["rank"],
                "career_id":       c["career_id"],
                "career_name":     c["career_name"],
                "emoji":           c["emoji"],
                "category":        c["category"],
                "stream":          c["stream"],
                "match_score":     c["match_score"],
                "confidence":      c["confidence"],
                "description":     c["description"],
                "required_skills": c["required_skills"][:5],
                "soft_skills":     c["soft_skills"][:4],
                "education_path":  c["education_path"][:4],
                "top_colleges":    c["top_colleges_india"][:4],
                "entrance_exams":  c["entrance_exams"][:3],
                "avg_salary_inr":  c["avg_salary_inr"],
                "avg_salary_global_usd": c["avg_salary_global_usd"],
                "future_demand":   c["future_demand"],
                "demand_growth":   f"+{c['demand_growth_pct']}%",
                "global_scope":    c["global_scope"],
                "job_market_note": c["job_market_note"],
                "riasec":          c["riasec"],
                "work_environment": c["work_environment"],
                "daily_tasks":     c["daily_tasks"],
                "suggested_next_skills": c["suggested_next_skills"][:5],
                "explanation":     c["explanation"],
            }
            for c in careers
        ],

        # ── Psychometric profile ──
        "skill_profile": {
            "scores":           profile["normalized_scores"],
            "dominant_traits":  profile["dominant_traits"],
            "secondary_traits": profile["secondary_traits"],
            "profile_type":     profile["profile_type"],
            "riasec_code":      profile["riasec_code"],
            "riasec_full":      profile["riasec_full"],
            "confidence":       profile["confidence"],
            "questions_answered": profile["questions_answered"],
        },

        # ── Market insights ──
        "market_insights": {
            "future_demand":      top.get("future_demand", "High"),
            "demand_growth_pct":  top.get("demand_growth_pct", 30),
            "avg_entry_salary":   top.get("avg_salary_inr", {}).get("entry", "N/A"),
            "global_scope":       top.get("global_scope", True),
            "job_market_note":    top.get("job_market_note", ""),
        },

        # ── Skill roadmap ──
        "suggested_skills": skills_list,

        # ── Summary card ──
        "summary": {
            "stream":           stream_name,
            "top_career":       top.get("career_name", ""),
            "top_career_emoji": top.get("emoji", ""),
            "top_match_pct":    top.get("match_score", 0),
            "profile_type":     profile["profile_type"],
            "riasec_code":      profile["riasec_code"],
            "riasec_full":      profile["riasec_full"],
            "analysis_quality": _quality(profile["confidence"]),
        },
    }


def _quality(cf: float) -> str:
    if cf >= 0.90: return "Comprehensive Analysis"
    if cf >= 0.70: return "Detailed Analysis"
    if cf >= 0.50: return "Standard Analysis"
    return "Preliminary Analysis"