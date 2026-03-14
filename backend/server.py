"""
LAKSHAY — Standalone HTTP Server
==================================
Runs the full backend with ONLY Python's built-in http.server.
No FastAPI / uvicorn / Flask required.

Usage
-----
    python server.py              →  http://localhost:8000
    python server.py --port 9000  →  http://localhost:9000

For production:
    pip install fastapi uvicorn
    uvicorn api.app:app --reload --port 8000
    Then visit http://localhost:8000/docs for interactive Swagger UI.
"""

import sys, os, json, argparse
from http.server  import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.questionnaire_engine import create_assessment, submit_answers, get_session, get_scored_answers
from core.scoring_engine        import calculate_profile
from core.recommendation_model  import train, model_status
from core.career_db             import get_career, all_careers, careers_by_stream
from core.analyzer              import run_analysis


# ── Bootstrap ──────────────────────────────────────────────────────────
print("━" * 55)
print("  🎯  LAKSHAY AI Career Guidance — Backend Server")
print("━" * 55)
print("  Bootstrapping ML model...")
result = train()
print(f"  Model status   : {result['status']}")
if "rf_cv_acc" in result:
    print(f"  RF  CV Accuracy: {result['rf_cv_acc']:.1%}")
    print(f"  KNN CV Accuracy: {result['knn_cv_acc']:.1%}")
    print(f"  DT  CV Accuracy: {result['dt_cv_acc']:.1%}")
print()


# ── HTTP Handler ────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"  [{self.command:6s}] {self.path:40s}  {args[1] if len(args)>1 else ''}")

    # ── CORS pre-flight ────────────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ── GET ────────────────────────────────────────────────────────
    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        qs     = parse_qs(parsed.query)

        if path == "/":
            return self._json({
                "service": "LAKSHAY AI Career Guidance API",
                "version": "3.0.0",
                "status":  "operational",
                "model":   model_status(),
                "endpoints": {
                    "POST /create-assessment":  "Start new assessment",
                    "POST /submit-answers":     "Record answers",
                    "POST /analyze-profile":    "Psychometric profile",
                    "POST /get-recommendations":"Full AI analysis",
                    "GET  /careers":            "Career knowledge base",
                    "GET  /careers/{id}":       "Single career",
                    "GET  /model/status":       "Model metrics",
                    "POST /model/train":        "Train model",
                }
            })

        if path == "/model/status":
            return self._json({"success": True, "data": model_status()})

        if path == "/careers":
            stream = qs.get("stream", [None])[0]
            data = careers_by_stream(stream) if stream else all_careers()
            return self._json({"success": True, "count": len(data), "data": data})

        if path.startswith("/careers/"):
            cid = path.split("/careers/")[-1]
            c   = get_career(cid)
            if c:
                return self._json({"success": True, "data": c})
            return self._json({"error": f"Career '{cid}' not found."}, 404)

        self._json({"error": f"Route not found: {path}"}, 404)

    # ── POST ───────────────────────────────────────────────────────
    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        body   = self._body()

        if path == "/create-assessment":
            user_type = body.get("user_type", "school_student")
            user_role = body.get("user_role", "student")
            return self._json({"success": True, "data": create_assessment(user_type, user_role)})

        if path == "/submit-answers":
            sid     = body.get("session_id")
            answers = body.get("answers", {})
            if not sid:
                return self._json({"error": "session_id required."}, 400)
            if not get_session(sid):
                return self._json({"error": "Session not found."}, 404)
            result = submit_answers(sid, answers)
            if "error" in result:
                return self._json({"error": result["error"]}, 400)
            return self._json({"success": True, "data": result})

        if path == "/analyze-profile":
            sid = body.get("session_id")
            if not sid:
                return self._json({"error": "session_id required."}, 400)
            scored = get_scored_answers(sid)
            if not scored:
                return self._json({"error": "Session not found."}, 404)
            profile = calculate_profile(scored["scored_answers"])
            return self._json({"success": True, "data": {"session_id": sid, "profile": profile}})

        if path == "/get-recommendations":
            sid = body.get("session_id")
            if not sid:
                return self._json({"error": "session_id required."}, 400)
            result = run_analysis(sid)
            if "error" in result:
                return self._json({"error": result["error"]}, 400)
            return self._json({"success": True, "data": result})

        if path == "/model/train":
            force  = body.get("force", False)
            result = train(force=force)
            return self._json({"success": True, "data": result})

        self._json({"error": f"Route not found: {path}"}, 404)

    # ── Helpers ────────────────────────────────────────────────────
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}


# ── Entry point ─────────────────────────────────────────────────────────

def run(port: int = 8000):
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"  🚀  Server running at  http://localhost:{port}")
    print(f"  📖  Health check:      http://localhost:{port}/")
    print(f"  📚  Careers:           http://localhost:{port}/careers")
    print()
    print("  Press  Ctrl+C  to stop.")
    print("━" * 55)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAKSHAY Backend Server")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    args = parser.parse_args()
    run(args.port)