"""
LAKSHAY — FastAPI Application
==============================
REST API layer. All business logic lives in core/.

Install & run
-------------
    pip install fastapi uvicorn
    cd lakshay_backend
    uvicorn api.app:app --reload --port 8000

Endpoints
---------
GET  /                        Health check
POST /create-assessment       Start a new assessment session
POST /submit-answers          Record question answers
POST /analyze-profile         Psychometric scoring only
POST /get-recommendations     Full AI analysis pipeline
GET  /careers                 All careers (optional ?stream= filter)
GET  /careers/{career_id}     Single career record
GET  /model/status            ML model metrics
POST /model/train             (Re)train the ML model

Interactive docs:  http://localhost:8000/docs
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

from core.questionnaire_engine import create_assessment, submit_answers, get_session, get_scored_answers
from core.scoring_engine        import calculate_profile
from core.recommendation_model  import train, model_status
from core.career_db             import get_career, all_careers, careers_by_stream
from core.analyzer              import run_analysis

if not _FASTAPI:
    print("FastAPI not installed.  Run: pip install fastapi uvicorn")
    print("Falling back to built-in HTTP server (server.py).")
else:

    app = FastAPI(
        title="LAKSHAY — AI Career Guidance API",
        description="Psychometric assessment + 3-model ML ensemble career recommendation for Indian students.",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Pydantic models ────────────────────────────────────────────────

    class CreateAssessmentReq(BaseModel):
        user_type: str = Field("school_student", description="school_student | college_student | college_graduate")
        user_role: str = Field("student",        description="student | parent")

        class Config:
            json_schema_extra = {"example": {"user_type": "school_student", "user_role": "student"}}

    class SubmitAnswersReq(BaseModel):
        session_id: str
        answers:    dict = Field(..., description='{"question_id": "option_id", ...}')

        class Config:
            json_schema_extra = {
                "example": {
                    "session_id": "your-uuid-here",
                    "answers":    {"ss_01": "A", "ss_02": "C", "ss_03": "B"}
                }
            }

    class SessionReq(BaseModel):
        session_id: str

    class TrainReq(BaseModel):
        force: bool = False

    # ── Routes ──────────────────────────────────────────────────────────

    @app.get("/", tags=["Health"])
    def health():
        return {
            "service": "LAKSHAY AI Career Guidance API",
            "version": "3.0.0",
            "status":  "operational",
            "model":   model_status(),
            "endpoints": {
                "POST /create-assessment":  "Start new assessment",
                "POST /submit-answers":     "Record answers (can batch)",
                "POST /analyze-profile":    "Psychometric profile only",
                "POST /get-recommendations":"Full AI analysis + explanations",
                "GET  /careers":            "Career knowledge base",
                "GET  /careers/{id}":       "Single career details",
            },
        }


    @app.post("/create-assessment", tags=["Assessment"])
    def create_assessment_ep(req: CreateAssessmentReq):
        """
        Start a new psychometric assessment session.
        Returns session_id, questions (no internal scores), and metadata.
        """
        data = create_assessment(req.user_type, req.user_role)
        return {"success": True, "data": data}


    @app.post("/submit-answers", tags=["Assessment"])
    def submit_answers_ep(req: SubmitAnswersReq):
        """
        Submit one or more answers. Can be called incrementally
        (answers accumulate per session).
        """
        session = get_session(req.session_id)
        if not session:
            raise HTTPException(404, "Session not found or expired.")

        result = submit_answers(req.session_id, req.answers)
        if "error" in result:
            raise HTTPException(400, result["error"])

        return {"success": True, "data": result}


    @app.post("/analyze-profile", tags=["Analysis"])
    def analyze_profile_ep(req: SessionReq):
        """
        Run psychometric scoring on submitted answers.
        Returns 7-dimensional profile WITHOUT career recommendations.
        """
        scored = get_scored_answers(req.session_id)
        if not scored:
            raise HTTPException(404, "Session not found or expired.")

        profile = calculate_profile(scored["scored_answers"])
        return {
            "success": True,
            "data": {
                "session_id": req.session_id,
                "profile":    profile,
            }
        }


    @app.post("/get-recommendations", tags=["Analysis"])
    def get_recommendations_ep(req: SessionReq):
        """
        Run the full AI pipeline:
        Scoring → ML prediction → Career enrichment → XAI explanations.
        Returns complete dashboard payload.
        """
        result = run_analysis(req.session_id)
        if "error" in result:
            raise HTTPException(400, result["error"])

        return {"success": True, "data": result}


    @app.get("/careers", tags=["Knowledge Base"])
    def list_careers(stream: str = Query(None, description="Filter by stream fragment, e.g. 'Technology'")):
        """List all careers. Optional stream filter."""
        data = careers_by_stream(stream) if stream else all_careers()
        return {"success": True, "count": len(data), "data": data}


    @app.get("/careers/{career_id}", tags=["Knowledge Base"])
    def career_detail(career_id: str):
        """Full details for a single career."""
        c = get_career(career_id)
        if not c:
            raise HTTPException(404, f"Career '{career_id}' not found.")
        return {"success": True, "data": c}


    @app.get("/model/status", tags=["Model"])
    def model_status_ep():
        return {"success": True, "data": model_status()}


    @app.post("/model/train", tags=["Model"])
    def train_ep(req: TrainReq):
        """Train or retrain the recommendation model."""
        result = train(force=req.force)
        return {"success": True, "data": result}