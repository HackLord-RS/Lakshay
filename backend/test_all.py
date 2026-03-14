"""
LAKSHAY — Complete Test Suite
================================
Tests all 6 modules end-to-end.

Run:  python tests/test_all.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.questionnaire_engine import create_assessment, submit_answers, get_scored_answers, get_session
from core.scoring_engine        import calculate_profile
from core.recommendation_model  import train, predict, model_status
from core.career_db             import get_career, all_careers, enrich
from core.explanation_engine    import explain, stream_rationale, skill_suggestions
from core.analyzer              import run_analysis

PASS = "✅"; FAIL = "❌"; SEP = "─" * 62

def hdr(t):
    print(f"\n{SEP}\n  {t}\n{SEP}")

def chk(label, cond, detail=""):
    s = PASS if cond else FAIL
    line = f"  {s}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return cond


# ══════════════════════════════════════════════════════════════════════
# 1. Questionnaire Engine
# ══════════════════════════════════════════════════════════════════════
def test_questionnaire():
    hdr("TEST 1 — Questionnaire Engine")
    ok = []

    # school student
    a = create_assessment("school_student", "student")
    ok.append(chk("create_assessment returns session_id",        "session_id" in a))
    ok.append(chk("15 questions served",                         a["total_questions"] == 15, f"got {a['total_questions']}"))
    ok.append(chk("Questions have no internal scores",
                   all("scores" not in opt
                       for q in a["questions"]
                       for opt in q["options"])))
    ok.append(chk("Each question has id, question, options",
                   all("id" in q and "question" in q and "options" in q
                       for q in a["questions"])))
    ok.append(chk("Options have id and text only",
                   all(set(opt.keys()) == {"id","text"}
                       for q in a["questions"] for opt in q["options"])))

    # parent
    p = create_assessment("school_student", "parent")
    ok.append(chk("Parent assessment created",                   "session_id" in p))
    ok.append(chk("Parent gets different session from student",  p["session_id"] != a["session_id"]))
    ok.append(chk("Parent gets 12 questions",                    p["total_questions"] == 12, f"got {p['total_questions']}"))

    # college student
    c = create_assessment("college_student")
    ok.append(chk("College student assessment created",          "session_id" in c))
    ok.append(chk("College student gets 12 questions",           c["total_questions"] == 12))

    # college graduate
    g = create_assessment("college_graduate")
    ok.append(chk("College graduate assessment created",         "session_id" in g))
    ok.append(chk("College graduate gets 10 questions",          g["total_questions"] == 10))

    # submit
    sid     = a["session_id"]
    answers = {q["id"]: q["options"][0]["id"] for q in a["questions"]}
    r       = submit_answers(sid, answers)
    ok.append(chk("submit_answers records all answers",          r["answered"] == 15))
    ok.append(chk("Status → completed after all answers",        r["status"] == "completed"))

    # scored
    scored = get_scored_answers(sid)
    ok.append(chk("get_scored_answers returns data",             scored is not None))
    ok.append(chk("Each scored answer has a scores dict",
                   all("scores" in s for s in scored["scored_answers"])))
    ok.append(chk("type_weight present in scored answers",
                   all("type_weight" in s for s in scored["scored_answers"])))

    return all(ok)


# ══════════════════════════════════════════════════════════════════════
# 2. Scoring Engine
# ══════════════════════════════════════════════════════════════════════
def test_scoring():
    hdr("TEST 2 — Psychometric Scoring Engine")
    ok = []

    a   = create_assessment("school_student", "student")
    sid = a["session_id"]
    # All A answers → analytical-biased profile
    submit_answers(sid, {q["id"]: "A" for q in a["questions"]})
    scored  = get_scored_answers(sid)
    profile = calculate_profile(scored["scored_answers"])

    ok.append(chk("Profile has normalized_scores",              "normalized_scores" in profile))
    ok.append(chk("7 dimensions present",                       len(profile["normalized_scores"]) == 7,
                   str(list(profile["normalized_scores"].keys()))))
    ok.append(chk("All scores in [10, 99]",
                   all(10 <= v <= 99 for v in profile["normalized_scores"].values()),
                   str(profile["normalized_scores"])))
    ok.append(chk("dominant_traits has 3 entries",              len(profile["dominant_traits"]) == 3))
    ok.append(chk("secondary_traits has 2 entries",             len(profile["secondary_traits"]) == 2))
    ok.append(chk("profile_type is non-empty string",           bool(profile["profile_type"])))
    ok.append(chk("riasec_code is 2–3 chars",                   2 <= len(profile["riasec_code"]) <= 3,
                   profile["riasec_code"]))
    ok.append(chk("riasec_full is non-empty",                   bool(profile["riasec_full"])))
    ok.append(chk("confidence in [0, 1]",                       0 <= profile["confidence"] <= 1))
    ok.append(chk("questions_answered == 15",                   profile["questions_answered"] == 15))

    print(f"\n   Scores:   {profile['normalized_scores']}")
    print(f"   Dominant: {profile['dominant_traits']}")
    print(f"   RIASEC:   {profile['riasec_code']}  ({profile['riasec_full']})")
    print(f"   Type:     {profile['profile_type']}")

    return all(ok)


# ══════════════════════════════════════════════════════════════════════
# 3. ML Model
# ══════════════════════════════════════════════════════════════════════
def test_model():
    hdr("TEST 3 — ML Career Recommendation Model (RF + KNN + DT)")
    ok = []

    # Train
    result = train(force=True)
    ok.append(chk("Model trains successfully",          result["status"] == "trained"))
    ok.append(chk("RF  CV accuracy ≥ 80%",              result.get("rf_cv_acc",0) >= 0.80,
                   f"{result.get('rf_cv_acc',0):.1%}"))
    ok.append(chk("KNN CV accuracy ≥ 75%",              result.get("knn_cv_acc",0) >= 0.75,
                   f"{result.get('knn_cv_acc',0):.1%}"))
    ok.append(chk("DT  CV accuracy ≥ 70%",              result.get("dt_cv_acc",0)  >= 0.70,
                   f"{result.get('dt_cv_acc',0):.1%}"))
    ok.append(chk("100 training samples",               result.get("n_samples",0) >= 100))
    ok.append(chk("20 career classes",                  result.get("n_classes",0) == 20))

    print(f"\n   RF  {result['rf_cv_acc']:.1%}  |  KNN {result['knn_cv_acc']:.1%}  |  DT {result['dt_cv_acc']:.1%}")

    # Predict analytical-technical
    preds = predict({
        "analytical":90,"technical":88,"creative":25,
        "social":20,"leadership":40,"business":30,"research":72
    }, top_n=5)
    ok.append(chk("Returns 5 predictions",              len(preds) == 5))
    ok.append(chk("All required keys present",
                   all({"rank","career_id","career_name","match_score","confidence"} <= set(p.keys())
                       for p in preds)))
    ok.append(chk("All match_scores in [10,98]",
                   all(10 <= p["match_score"] <= 98 for p in preds)))
    ok.append(chk("Rank-1 is a tech/research career",
                   preds[0]["career_id"] in {
                       "data_scientist","ai_ml_engineer","software_engineer",
                       "research_scientist","robotics_engineer"
                   }, f"got: {preds[0]['career_id']} {preds[0]['match_score']}%"))

    print(f"\n   Analytical-Technical top picks:")
    for p in preds[:3]:
        print(f"     {p['rank']}. {p['career_name']:40s} {p['match_score']:3d}%  [{p['confidence']}]")

    # Predict creative-social
    cpreds = predict({
        "analytical":28,"technical":20,"creative":90,
        "social":68,"leadership":52,"business":45,"research":22
    }, top_n=5)
    ok.append(chk("Creative-Social top career is design/arts/psychology",
                   cpreds[0]["career_id"] in {
                       "ux_ui_designer","architect","clinical_psychologist",
                       "fashion_designer","journalist","teacher_educator"
                   }, f"got: {cpreds[0]['career_id']}"))

    print(f"\n   Creative-Social top picks:")
    for p in cpreds[:3]:
        print(f"     {p['rank']}. {p['career_name']:40s} {p['match_score']:3d}%  [{p['confidence']}]")

    return all(ok)


# ══════════════════════════════════════════════════════════════════════
# 4. Career Knowledge Database
# ══════════════════════════════════════════════════════════════════════
def test_career_db():
    hdr("TEST 4 — Career Knowledge Database")
    ok = []

    careers = all_careers()
    ok.append(chk("Database loads",                         len(careers) > 0))
    ok.append(chk("20 careers present",                     len(careers) == 20, f"found {len(careers)}"))

    required_fields = [
        "id","name","stream","required_skills","avg_salary_inr",
        "future_demand","education_path","profile_weights","riasec",
        "suggested_next_skills","daily_tasks","work_environment"
    ]
    for f in required_fields:
        ok.append(chk(f"All careers have '{f}'",            all(f in c for c in careers)))

    ds = get_career("data_scientist")
    ok.append(chk("data_scientist career found",            ds is not None))
    ok.append(chk("data_scientist has salary entry-level",  "entry" in ds.get("avg_salary_inr",{})))
    ok.append(chk("data_scientist has daily tasks",         len(ds.get("daily_tasks",[])) > 0))

    # Enrichment
    preds = [
        {"rank":1,"career_id":"data_scientist","career_name":"Data Scientist","match_score":92,"confidence":"Very High","raw_prob":0.5},
        {"rank":2,"career_id":"ux_ui_designer","career_name":"UX/UI","match_score":55,"confidence":"Moderate","raw_prob":0.2},
    ]
    enriched = enrich(preds)
    ok.append(chk("Enrichment adds description",            all("description" in e for e in enriched)))
    ok.append(chk("Enrichment adds job_market_note",        all("job_market_note" in e for e in enriched)))
    ok.append(chk("Enrichment adds emoji",                  all("emoji" in e for e in enriched)))
    ok.append(chk("Enrichment preserves match_score",       enriched[0]["match_score"] == 92))

    return all(ok)


# ══════════════════════════════════════════════════════════════════════
# 5. Explanation Engine
# ══════════════════════════════════════════════════════════════════════
def test_explanations():
    hdr("TEST 5 — Explanation Engine (XAI)")
    ok = []

    sample_profile = {
        "normalized_scores": {
            "analytical":88,"technical":82,"creative":30,
            "social":25,"leadership":45,"business":35,"research":72
        },
        "dominant_traits":  ["analytical","technical","research"],
        "secondary_traits": ["leadership","creative"],
        "profile_type":     "Analytical–Technical",
        "riasec_code":      "IR",
        "riasec_full":      "Investigative + Realistic",
        "confidence":       0.95,
    }

    exp = explain("data_scientist","Data Scientist", sample_profile, rank=1)
    ok.append(chk("Explanation has 'short'",                "short" in exp))
    ok.append(chk("Explanation has 'detailed'",             "detailed" in exp))
    ok.append(chk("Explanation has 'key_reasons'",          len(exp.get("key_reasons",[])) > 0))
    ok.append(chk("'short' is non-empty",                   len(exp["short"]) > 20))
    ok.append(chk("'key_reasons' are all strings",          all(isinstance(r,str) for r in exp["key_reasons"])))
    ok.append(chk("'fit_statement' present",                "fit_statement" in exp))

    rat = stream_rationale("Science – Technology", sample_profile)
    ok.append(chk("Stream rationale is non-empty paragraph", len(rat) > 80))

    print(f"\n   Short:   {exp['short'][:100]}...")
    print(f"   Reason:  {exp['key_reasons'][0][:100]}...")
    if exp.get("caution"):
        print(f"   Caution: {exp['caution'][:100]}...")

    return all(ok)


# ══════════════════════════════════════════════════════════════════════
# 6. Full Pipeline (End-to-End)
# ══════════════════════════════════════════════════════════════════════
def test_full_pipeline():
    hdr("TEST 6 — Full Analysis Pipeline (End-to-End)")
    ok = []

    # Create school student session
    a   = create_assessment("school_student", "student")
    sid = a["session_id"]

    # Mixed answers — analytical/research bias
    answers = {}
    for i, q in enumerate(a["questions"]):
        answers[q["id"]] = "A" if i % 2 == 0 else "B"
    submit_answers(sid, answers)

    # Run full analysis
    result = run_analysis(sid)

    ok.append(chk("run_analysis returns no error",              "error" not in result))
    ok.append(chk("Has recommended_stream",                     "recommended_stream" in result))
    ok.append(chk("Has stream_rationale",                       bool(result.get("stream_rationale",""))))
    ok.append(chk("Has top_careers (5 entries)",                len(result.get("top_careers",[])) == 5))
    ok.append(chk("Has skill_profile",                          "skill_profile" in result))
    ok.append(chk("Has market_insights",                        "market_insights" in result))
    ok.append(chk("Has suggested_skills (≥4)",                  len(result.get("suggested_skills",[])) >= 4))
    ok.append(chk("Has summary block",                          "summary" in result))
    ok.append(chk("Summary has riasec_code",                    "riasec_code" in result["summary"]))

    top = result["top_careers"][0]
    for f in ["career_id","career_name","emoji","match_score","confidence",
              "required_skills","education_path","avg_salary_inr",
              "future_demand","demand_growth","global_scope","explanation",
              "daily_tasks","work_environment","suggested_next_skills"]:
        ok.append(chk(f"Top career has '{f}'",                  f in top))

    ok.append(chk("Top career explanation has 'short'",         "short" in top.get("explanation",{})))
    ok.append(chk("Top career explanation has 'key_reasons'",   len(top.get("explanation",{}).get("key_reasons",[])) > 0))

    print(f"\n  ✦  Result Summary:")
    print(f"     Stream         : {result['recommended_stream']}")
    print(f"     Top Career     : {top['emoji']} {top['career_name']}  ({top['match_score']}%)")
    print(f"     Profile Type   : {result['skill_profile']['profile_type']}")
    print(f"     RIASEC         : {result['skill_profile']['riasec_code']}  ({result['skill_profile']['riasec_full']})")
    print(f"     Confidence     : {result['skill_profile']['confidence']:.0%}")
    print(f"     Analysis Quality: {result['summary']['analysis_quality']}")
    print(f"\n     All 5 careers:")
    for c in result["top_careers"]:
        print(f"       {c['rank']}. {c['emoji']} {c['career_name']:42s} {c['match_score']:3d}%")
    print(f"\n     Suggested skills: {', '.join(result['suggested_skills'][:5])}")

    return all(ok)


# ══════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*62)
    print("  LAKSHAY Backend — Full Test Suite")
    print("═"*62)

    suites = [
        ("Questionnaire Engine",    test_questionnaire),
        ("Psychometric Scoring",    test_scoring),
        ("ML Model (3-model ensemble)", test_model),
        ("Career Knowledge DB",     test_career_db),
        ("Explanation Engine",      test_explanations),
        ("Full Pipeline",           test_full_pipeline),
    ]

    passed = 0
    for name, fn in suites:
        try:
            if fn():
                passed += 1
        except Exception as e:
            import traceback
            print(f"\n  {FAIL}  {name} — EXCEPTION: {e}")
            traceback.print_exc()

    print(f"\n{'═'*62}")
    print(f"  RESULT: {passed}/{len(suites)} test suites passed")
    print(f"{'═'*62}\n")
    return passed == len(suites)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)