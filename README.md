# Medical Triage Agent

A comprehensive README for the [medical_triage_agent](https://github.com/vedant-maheshwari/medical_triage_agent) – a FastAPI-based medical triage platform with AI-powered symptom analysis, learning from doctor feedback, visual wound/image assessment, and an analytics dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
  - [Install Dependencies](#install-dependencies)
  - [Setup Environment](#setup-environment)
  - [Start MongoDB](#start-mongodb)
  - [Run the API Server](#run-the-api-server)
  - [Optional: Streamlit UI](#optional-streamlit-ui)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
  - [Patient Portal](#patient-portal)
  - [Doctor Review](#doctor-review)
  - [Analytics Endpoints](#analytics-endpoints)
  - [Admin Endpoints](#admin-endpoints)
- [Testing & Examples](#testing--examples)
- [Troubleshooting](#troubleshooting)
- [License & Contributing](#license--contributing)
- [Resources](#resources)

---

## Overview

This project provides an AI-powered medical triage API that leverages GPT-4o (with vision) to:
- Analyze patient symptoms and descriptions
- Assess images (e.g., wounds, visible conditions)
- Learn from doctor corrections to improve over time
- Provide analytics for continuous AI performance tracking

---

## Features

- 🤖 **AI-Powered Triage**: Analyze symptoms and suggest urgency/next steps
- 📸 **Image Analysis**: Accepts patient/wound images for automated assessment
- 👨‍⚕️ **Doctor Review Flow**: Cases routed for doctor validation and correction
- 🧠 **Continuous Learning**: AI model retrains from physician feedback, leveraging embeddings and pattern detection
- 📊 **Analytics Dashboard**: Track triage accuracy, review outcomes, and AI improvement
- 🔧 **Admin Tools**: System stats, cache clearing, model prompt/context review, and more

---

## Quick Start

### 1. Install Dependencies

pip install -r requirements.txt

### 2. Setup Environment

cp .env.template .env
# Edit .env and add your OPENAI_API_KEY and MongoDB connection details

#### .env file:

OPENAI_API_KEY=sk-your-key-here
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=medical_triage
TRIAGE_MODEL=gpt-4o
DEBUG=false

### 3. Start MongoDB

**Using Docker**:
docker run -d -p 27017:27017 --name mongodb mongo:latest
**Or on macOS (Homebrew)**:
brew services start mongodb-community

### 4. Run the API Server

uvicorn app.main:app --reload
API available at [http://127.0.0.1:8000](http://127.0.0.1:8000)

### 5. (Optional) Streamlit UI

streamlit run streamlit_ui.py
UI will be available at [http://localhost:8501](http://localhost:8501)

---

## Environment Variables

| Variable            | Required | Default                         | Description                     |
|---------------------|:--------:|---------------------------------|---------------------------------|
| OPENAI_API_KEY      | Yes*     | -                               | OpenAI GPT-4o key for analysis  |
| MONGODB_URI         | No       | mongodb://localhost:27017/      | MongoDB connection string       |
| MONGODB_DB          | No       | medical_triage                  | Database name                   |
| TRIAGE_MODEL        | No       | gpt-4o                          | AI model used                   |
| DEBUG               | No       | false                           | Debug mode for development      |

*`OPENAI_API_KEY` is needed for AI-based endpoints.

---

## Project Structure

wound_triage_api/
├── app/
│   ├── __init__.py
│   ├── main.py         # FastAPI app and endpoints
│   ├── services.py     # AI agent logic
│   ├── models.py       # Pydantic models
│   ├── config.py       # Settings and config
│   └── database.py     # MongoDB connector
├── streamlit_ui.py     # Streamlit client UI
├── full_app.py         # Reference Streamlit app
├── requirements.txt    # Python dependencies
├── .env.template       # Env variable template
└── test_api.sh         # Automated API tests

---

## API Documentation

### Interactive Docs

- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### Patient Portal

Submit symptoms and images for triage.

#### **POST /triage/assess**

Submit new triage request for AI analysis.

**Request** (multipart/form-data):

| Field         | Type    | Required | Description                          |
|---------------|---------|----------|--------------------------------------|
| description   | string  | Yes      | Patient symptom/case description     |
| patient_id    | string  | Yes      | Patient identifier                   |
| image         | file    | No       | Photo of wound/condition (optional)  |

**Example (cURL):**
curl -X POST "http://127.0.0.1:8000/triage/assess" \
  -F "description=Severe chest pain with shortness of breath" \
  -F "patient_id=TEST_001"

**Response** (JSON):
{
  "case_id": "UUID",
  "ai_assessment": "description/urgency",
  "confidence": 0.94,
  "triage_level": "Emergency/Consult/Manage at Home",
  "comments": "AI notes"
}

#### **GET /triage/case/{case_id}**

Get details for a given triage case.

#### **GET /triage/case/{case_id}/image**

Retrieve the image for a specific case if provided.

---

### Doctor Review

For medical reviewers to validate or correct AI assessments.

#### **GET /triage/pending-reviews**

Retrieve a list of triage cases pending physician review.

**Response** (JSON):
[
  {"case_id": "...", "description": "...", "triage_level": "AI suggestion", ...}
]

#### **POST /triage/validate**

Submit a doctor's assessment/correction for a triage case.

**Request** (JSON):
{
  "case_id": "string",
  "doctor_id": "string",
  "corrected_triage_level": "string",
  "doctor_comments": "string"
}

**Response**:
- `200 OK` on success; includes updated case info.

---

### Analytics Endpoints

Track system stats and AI learning progress.

#### **GET /triage/analytics**

Returns analytics data, model performance, feedback summary, etc.

#### **GET /triage/recent-cases**

Get a list of the most recent triage cases for review/stats.

---

### Admin Endpoints

System and model management (caution: privileged operations).

#### **GET /admin/stats**

Current system-wide statistics.

#### **POST /admin/clear-cache**

Force clear AI pattern/model cache.

#### **POST /admin/reset-system**

Reset internal model/state (for maintenance/testing).

#### **POST /admin/recalculate-scores**

Recompute model accuracy and learning metrics.

#### **GET /admin/prompt**

Fetch the current AI prompt template in use.

#### **GET /admin/pattern-context**

Retrieve contextual knowledge/patterns used by the model.

---

## Testing & Examples

### Health Check

curl http://127.0.0.1:8000/health

### API Request Example

See [`CURL_EXAMPLES.md`](CURL_EXAMPLES.md) and [`QUICK_TEST.md`](QUICK_TEST.md) for complete sample requests and advanced usage.

---

## Troubleshooting

**"OPENAI_API_KEY not set"**  
- Create `.env` from `.env.template`  
- Add your OpenAI API key  
- Restart the server

**"MongoDB connection failed"**  
- Confirm MongoDB is running  
- Check `MONGODB_URI` in `.env`

**Import errors**  
- Reinstall dependencies:  
  pip install -r requirements.txt

---

## License & Contributing

This project is for educational/demo purposes.  
**For production use:**  
- Add strong authentication and authorization
- Comply with healthcare data regulations (HIPAA/GDPR)
- Perform a security review (see `CONTRIBUTING.md` for guidelines)

PRs and issues are welcome for educational collaboration!

---

## Resources

- [Setup Guide](SETUP.md)
- [API Curl Examples](CURL_EXAMPLES.md)
- [Quick Test Reference](QUICK_TEST.md)

**GitHub:** [vedant-maheshwari/medical_triage_agent](https://github.com/vedant-maheshwari/medical_triage_agent)

---