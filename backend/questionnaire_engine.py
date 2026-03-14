"""
LAKSHAY — Questionnaire Engine
=================================
Manages assessment sessions and serves scientifically grounded
psychometric questions without exposing internal scoring data.

Scientific grounding:
  • Holland RIASEC Model (1973) — interest mapping
  • Situational Judgment Tests — behavioural prediction
  • Big Five adaptation — personality traits
  • Kolb Learning Styles — cognitive preference
"""

import json
import uuid
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

# ── In-memory session store (replace with Redis in production) ─────────
_sessions: dict[str, dict] = {}


def _load_question_bank() -> dict:
    with open(DATA_DIR / "questions.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ── Public API ─────────────────────────────────────────────────────────

def create_assessment(user_type: str, user_role: str = "student") -> dict:
    """
    Start a new psychometric assessment session.

    Parameters
    ----------
    user_type : str  school_student | college_student | college_graduate
    user_role : str  student | parent  (relevant for school_student)

    Returns
    -------
    dict  {session_id, user_type, user_role, total_questions, sections, questions[]}
          — questions have NO internal scoring data (stripped before return)
    """
    bank = _load_question_bank()

    # Route to correct question set
    if user_type == "school_student" and user_role == "parent":
        key = "parent"
    elif user_type in bank and user_type not in ("_meta",):
        key = user_type
    else:
        key = "school_student"

    question_set = bank[key]
    session_id = str(uuid.uuid4())

    # Strip internal scoring before sending to client
    clean_questions = [
        {
            "id": q["id"],
            "section": q.get("section", ""),
            "type": q.get("type", ""),
            "question": q["question"],
            "hint": q.get("hint", ""),
            "options": [{"id": o["id"], "text": o["text"]} for o in q["options"]],
        }
        for q in question_set["questions"]
    ]

    _sessions[session_id] = {
        "session_id":     session_id,
        "user_type":      user_type,
        "user_role":      user_role,
        "question_key":   key,
        "total":          len(clean_questions),
        "answers":        {},   # {question_id: chosen_option_id}
        "status":         "in_progress",
    }

    return {
        "session_id":      session_id,
        "user_type":       user_type,
        "user_role":       user_role,
        "label":           question_set.get("label", ""),
        "description":     question_set.get("description", ""),
        "total_questions": len(clean_questions),
        "sections":        question_set.get("sections", []),
        "questions":       clean_questions,
    }


def submit_answers(session_id: str, answers: dict) -> dict:
    """
    Record answers for a session. Can be called multiple times —
    answers accumulate and later calls overwrite earlier ones for the same question.

    Parameters
    ----------
    answers : dict  {question_id: option_id}  e.g. {"ss_01": "A", "ss_02": "C"}

    Returns
    -------
    dict  {session_id, answered, total, status}
    """
    session = _sessions.get(session_id)
    if not session:
        return {"error": "Session not found or expired."}

    session["answers"].update(answers)
    n_answered = len(session["answers"])

    if n_answered >= session["total"]:
        session["status"] = "completed"

    return {
        "session_id":  session_id,
        "answered":    n_answered,
        "total":       session["total"],
        "status":      session["status"],
    }


def get_scored_answers(session_id: str) -> Optional[dict]:
    """
    Retrieve answers enriched with their internal scoring data.
    Called by the scoring engine — never exposed to the client.

    Returns
    -------
    dict  {session_id, user_type, user_role, scored_answers: [...]}
    """
    session = _sessions.get(session_id)
    if not session:
        return None

    bank = _load_question_bank()
    question_list = bank[session["question_key"]]["questions"]
    q_map = {q["id"]: q for q in question_list}

    # Get section_type weights from meta
    type_weights = bank.get("_meta", {}).get("section_weights", {})

    scored = []
    for qid, opt_id in session["answers"].items():
        q = q_map.get(qid)
        if not q:
            continue
        opt = next((o for o in q["options"] if o["id"] == opt_id), None)
        if opt:
            q_type = q.get("type", "interest_mapping")
            scored.append({
                "question_id":  qid,
                "section":      q.get("section", ""),
                "type":         q_type,
                "type_weight":  type_weights.get(q_type, 1.0),
                "option_id":    opt_id,
                "scores":       opt["scores"],
            })

    return {
        "session_id":    session_id,
        "user_type":     session["user_type"],
        "user_role":     session["user_role"],
        "scored_answers": scored,
        "answer_count":  len(scored),
    }


def get_session(session_id: str) -> Optional[dict]:
    """Return session metadata (no scoring data)."""
    s = _sessions.get(session_id)
    if not s:
        return None
    return {
        "session_id": s["session_id"],
        "user_type":  s["user_type"],
        "user_role":  s["user_role"],
        "answered":   len(s["answers"]),
        "total":      s["total"],
        "status":     s["status"],
    }