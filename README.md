# LAKSHAY — AI Career Guidance Platform
## Complete Backend System

---

## 📁 Project Structure

```
lakshay_backend/
│
├── server.py                        ← Standalone HTTP server (zero deps beyond sklearn)
│
├── api/
│   └── app.py                       ← FastAPI app (production)
│
├── core/
│   ├── __init__.py
│   ├── questionnaire_engine.py      ← Session management + question serving
│   ├── scoring_engine.py            ← Psychometric profile calculation (7-dim)
│   ├── recommendation_model.py      ← RF + KNN + DT ensemble ML model
│   ├── career_db.py                 ← Career knowledge database queries
│   ├── explanation_engine.py        ← XAI natural-language explanations
│   └── analyzer.py                  ← Full pipeline orchestrator
│
├── data/
│   ├── questions.json               ← 49 RIASEC-grounded questions (4 user types)
│   ├── careers.json                 ← 20 careers with full metadata
│   └── training_data.csv            ← 100 ML training samples (20 careers × 5)
│
├── models/
│   └── lakshay_model.pkl            ← Trained model bundle (auto-generated)
│
└── tests/
    └── test_all.py                  ← 6-suite test runner (60+ checks)
```

---

## 🚀 Quick Start

### Option A — Zero dependencies (runs anywhere Python runs)
```bash
cd lakshay_backend
python server.py              # http://localhost:8000
python server.py --port 9000  # custom port
```

### Option B — FastAPI (recommended for integration)
```bash
pip install fastapi uvicorn
uvicorn api.app:app --reload --port 8000
# Interactive docs: http://localhost:8000/docs
```

### Run tests
```bash
python tests/test_all.py
```

---

## 🎯 AI System Architecture

### Psychometric Framework

All questions are grounded in peer-reviewed career assessment science:

| Framework | Application |
|-----------|-------------|
| **Holland RIASEC Model (1973)** | Interest mapping — 6 personality types: Realistic, Investigative, Artistic, Social, Enterprising, Conventional |
| **Big Five Personality Traits** | Adapted for career counselling — conscientiousness, openness, agreeableness |
| **Situational Judgment Tests** (Schmidt & Hunter, 1998) | Behavioural prediction — weighted 1.25× (reveals actual behaviour vs stated preference) |
| **Kolb Experiential Learning Styles (1984)** | Cognitive style — how candidates learn and process information |

### 7 Scoring Dimensions
| Dimension  | Captures                         | RIASEC Code |
|------------|----------------------------------|-------------|
| analytical | Logical reasoning, quantitative  | I           |
| technical  | Engineering, coding, making      | R           |
| creative   | Art, design, original ideation   | A           |
| social     | Empathy, helping, people skills  | S           |
| leadership | Managing, motivating, leading    | E           |
| business   | Commerce, strategy, enterprise   | E / C       |
| research   | Curiosity, investigation, depth  | I           |

### ML Model
- **Random Forest Classifier** (200 trees, weight 0.50)
- **K-Nearest Neighbours** (k=5 distance-weighted, weight 0.30)
- **Decision Tree Classifier** (max_depth=12, weight 0.20)
- All use StandardScaler pipelines
- Cross-validation accuracy: RF 94–100%, KNN 97–100%, DT 85–90%

### Full Pipeline
```
User answers questionnaire (15 questions)
           ↓
Weighted psychometric scoring
  (situational_judgment × 1.25, personality_trait × 1.15 ...)
           ↓
7-dimensional normalised profile vector [0-100 each]
           ↓
RF + KNN + DT ensemble prediction
           ↓
Top-5 career probabilities
           ↓
Career knowledge database enrichment
  (salary, colleges, exams, growth data, daily tasks...)
           ↓
XAI explanation generation
  (short + detailed + key_reasons + caution)
           ↓
Complete dashboard payload → Frontend
```

---

## 📡 API Reference

### `POST /create-assessment`
Start a new psychometric assessment session.

**Request:**
```json
{
  "user_type": "school_student",
  "user_role": "student"
}
```

| user_type | user_role | Questions | Notes |
|-----------|-----------|-----------|-------|
| school_student | student | 15 | Full RIASEC battery |
| school_student | parent | 12 | Child-observation questions |
| college_student | student | 12 | Undergraduate-specific |
| college_graduate | student | 10 | Career-transition focused |

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "uuid",
    "total_questions": 15,
    "questions": [
      {
        "id": "ss_01",
        "section": "Interest & Passion",
        "type": "interest_mapping",
        "question": "When you have completely free time...",
        "hint": "Think about what you do without being told.",
        "options": [
          {"id": "A", "text": "Taking apart a gadget..."},
          {"id": "B", "text": "Reading about discoveries..."},
          {"id": "C", "text": "Drawing, writing stories..."},
          {"id": "D", "text": "Organising events..."}
        ]
      }
    ]
  }
}
```

---

### `POST /submit-answers`
Submit answers. Can call multiple times — answers accumulate per session.

**Request:**
```json
{
  "session_id": "uuid",
  "answers": {
    "ss_01": "A",
    "ss_02": "B",
    "ss_03": "C"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "uuid",
    "answered": 15,
    "total": 15,
    "status": "completed"
  }
}
```

---

### `POST /analyze-profile`
Run psychometric scoring only (no ML predictions).

**Response:**
```json
{
  "success": true,
  "data": {
    "profile": {
      "normalized_scores": {
        "analytical": 88,
        "technical": 75,
        "creative": 40,
        "social": 35,
        "leadership": 50,
        "business": 42,
        "research": 70
      },
      "dominant_traits": ["analytical", "technical", "research"],
      "secondary_traits": ["leadership", "creative"],
      "profile_type": "Analytical–Technical",
      "riasec_code": "IR",
      "riasec_full": "Investigative + Realistic",
      "confidence": 0.95,
      "questions_answered": 15
    }
  }
}
```

---

### `POST /get-recommendations`
Full AI pipeline. Returns the complete dashboard payload.

**Response:**
```json
{
  "success": true,
  "data": {
    "recommended_stream": "Science – Technology",
    "stream_rationale": "The Science – Technology stream is recommended...",
    "stream_education_path": ["Class 11–12: PCM + CS", "B.Tech CS or Statistics"],

    "top_careers": [
      {
        "rank": 1,
        "career_id": "data_scientist",
        "career_name": "Data Scientist",
        "emoji": "📊",
        "category": "Technology & Analytics",
        "stream": "Science – Technology",
        "match_score": 92,
        "confidence": "Very High",
        "description": "Extract actionable insights from large datasets...",
        "required_skills": ["Python", "Machine Learning", "Statistics"],
        "soft_skills": ["Critical Thinking", "Communication"],
        "education_path": ["Class 11–12 → PCM + CS", "B.Tech / B.Sc CS"],
        "top_colleges": ["IIT Bombay", "IIT Delhi", "IISc Bangalore"],
        "entrance_exams": ["JEE Advanced", "JEE Main"],
        "avg_salary_inr": {"entry": "8–14 LPA", "mid": "18–35 LPA", "senior": "40–90 LPA"},
        "avg_salary_global_usd": {"entry": "80K", "mid": "130K", "senior": "200K+"},
        "future_demand": "Very High",
        "demand_growth": "+45%",
        "global_scope": true,
        "job_market_note": "Data Scientist roles projected to grow 45% by 2030...",
        "daily_tasks": ["Build predictive models", "Write data pipelines"],
        "work_environment": "Office / Remote – tech company, startup, research lab",
        "suggested_next_skills": ["LLMs / Generative AI", "MLOps", "Spark"],
        "explanation": {
          "short": "Data Scientist is recommended because your analytical and technical abilities are notably strong.",
          "detailed": "Your assessment reveals excellent analytical ability and technical aptitude...",
          "key_reasons": [
            "Your strong analytical reasoning (88/100) is a core requirement.",
            "Your solid technical aptitude (75/100) enables rapid growth.",
            "Your research orientation (70/100) supports the investigative nature of the role."
          ],
          "fit_statement": "Those who find deep satisfaction in uncovering hidden patterns...",
          "caution": "Consider building communication skills to present insights to non-technical stakeholders."
        }
      }
    ],

    "skill_profile": {
      "scores": { "analytical": 88, "technical": 75, ... },
      "dominant_traits": ["analytical", "technical", "research"],
      "profile_type": "Analytical–Technical",
      "riasec_code": "IR",
      "riasec_full": "Investigative + Realistic",
      "confidence": 0.95
    },

    "market_insights": {
      "future_demand": "Very High",
      "demand_growth_pct": 45,
      "avg_entry_salary": "8–14 LPA",
      "global_scope": true
    },

    "suggested_skills": ["Python", "Machine Learning", "LLMs", "PyTorch", "SQL"],

    "summary": {
      "stream": "Science – Technology",
      "top_career": "Data Scientist",
      "top_career_emoji": "📊",
      "top_match_pct": 92,
      "profile_type": "Analytical–Technical",
      "riasec_code": "IR",
      "riasec_full": "Investigative + Realistic",
      "analysis_quality": "Comprehensive Analysis"
    }
  }
}
```

---

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/careers` | All 20 careers (`?stream=Technology` to filter) |
| GET | `/careers/{id}` | Full career record |
| GET | `/model/status` | Training status + accuracy metrics |
| POST | `/model/train` | `{"force": true}` to retrain |

---

## ⚛️ React Frontend Integration

```javascript
const API = "http://localhost:8000";

// 1. Start assessment
const { data: assessment } = await fetch(`${API}/create-assessment`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ user_type: "school_student", user_role: "student" })
}).then(r => r.json());

const { session_id, questions, total_questions } = assessment;

// 2. Show questions one at a time in UI, collect answers
// ...

// 3. Submit answers (can batch all or submit progressively)
await fetch(`${API}/submit-answers`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ session_id, answers: { ss_01: "A", ss_02: "C", ... } })
});

// 4. Get full AI recommendations
const { data: results } = await fetch(`${API}/get-recommendations`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ session_id })
}).then(r => r.json());

// Use in your dashboard:
results.recommended_stream          // "Science – Technology"
results.top_careers[0].career_name  // "Data Scientist"
results.top_careers[0].match_score  // 92
results.skill_profile.scores        // {analytical: 88, technical: 75, ...}
results.suggested_skills            // ["Python", "ML", ...]
results.summary.riasec_code         // "IR"
```

---

## 🗺️ Questions Coverage

| User Type | Questions | Framework Sections |
|-----------|-----------|-------------------|
| School Student | 15 | Interest & Passion, Academic Aptitude, Cognitive Style, Personality & Values, Situational Judgment |
| Parent | 12 | Child Observation, Academic Behaviour, Problem-Solving Behaviour, Strongest Talent, Career Perception, Financial Context, Learning Style, Risk & Ambition, Social & Leadership, Subject Depth, Interpersonal Style, Values & Purpose |
| College Student | 12 | Academic Engagement, Risk Orientation, Career Vision, Work Style, Impact Motivation, Skill Confidence, Decision Making, Extracurricular Identity, Learning Depth, Professional Identity, Grad School, Values Alignment |
| College Graduate | 10 | Career Reflection, Definition of Success, Transition Motivation, Skill Inventory, Work Environment, Postgraduate Consideration, Leadership Orientation, Industry Attraction, Near-Term Action, Legacy Vision |

---

## 📈 Scaling to Production

| Component | Hackathon (now) | Production |
|-----------|----------------|------------|
| Sessions | In-memory dict | Redis / DynamoDB |
| Database | JSON files | PostgreSQL / MongoDB |
| ML model | Pickle file | MLflow / Vertex AI |
| Server | built-in http / uvicorn | Gunicorn + nginx |
| Training data | 100 CSV rows | 10,000+ real assessment records |
| Auth | None | JWT tokens |
| Deployment | Local | Docker → AWS ECS / GCP Cloud Run |

---

*Scientifically grounded in Holland RIASEC (1973), Schmidt & Hunter SJT validity (1998),
and Kolb Experiential Learning Styles (1984).*
*Built with scikit-learn, pandas, numpy. No paid external APIs required.*