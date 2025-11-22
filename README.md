# Medical Triage AI API

A sophisticated FastAPI-based medical triage system with AI-powered symptom analysis, image processing, and continuous learning capabilities. This system leverages GPT-4o with vision support to provide intelligent patient assessment while enabling doctors to refine and improve AI predictions over time.

---

## Features

- **AI-Powered Triage**: Analyze patient symptoms using GPT-4o with advanced vision capabilities
- **Image Analysis**: Upload and analyze images of wounds and medical conditions
- **Doctor Review System**: Medical professionals can review, validate, and correct AI assessments
- **Continuous Learning**: AI learns from doctor corrections using embeddings and pattern analysis
- **Analytics Dashboard**: Real-time tracking of AI performance and learning progress
- **Admin Tools**: Comprehensive system management and maintenance utilities
- **Scalable Architecture**: MongoDB-backed persistent storage with RESTful API design

---

## Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose (optional)
- MongoDB
- OpenAI API Key

### 1. Clone Repository

```bash
git clone https://github.com/vedant-maheshwari/medical_triage_agent.git
cd medical_triage_agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables

```bash
# Copy the template
cp .env.template .env

# Edit .env with your values
nano .env
```

**Required variables:**
```
OPENAI_API_KEY=sk-your-api-key-here
MONGODB_URI=mongodb://localhost:27017/
```

### 4. Start MongoDB

**Option A: Using Docker**
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

**Option B: Using Homebrew (macOS)**
```bash
brew services start mongodb-community
```

### 5. Start the API Server

```bash
uvicorn app.main:app --reload
```

The API will be available at:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **API Base URL**: http://127.0.0.1:8000

### 6. (Optional) Start Streamlit UI

```bash
streamlit run streamlit_ui.py
```

UI will be available at: http://localhost:8501

---

## Project Structure

```
medical_triage_agent/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application & endpoints
│   ├── services.py          # Business logic & AI agent
│   ├── models.py            # Pydantic data models
│   ├── config.py            # Configuration settings
│   └── database.py          # MongoDB connection handler
├── streamlit_ui.py          # Streamlit frontend interface
├── full_app.py              # Original Streamlit app (reference)
├── requirements.txt         # Python dependencies
├── .env.template            # Environment variables template
├── test_api.sh              # Automated API test script
└── README.md                # This file
```

---

## API Endpoints Documentation

### Health Check

#### `GET /health`

Health check endpoint - no authentication required.

**Request:**
```bash
curl http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-11-22T21:00:00Z"
}
```

---

### Patient Portal

#### `POST /triage/assess`

Submit patient symptoms for AI-powered triage assessment.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/triage/assess" \
  -F "description=Severe chest pain with shortness of breath" \
  -F "patient_id=PATIENT_001"
```

**Request Body:**
```
Content-Type: multipart/form-data

- description (string, required): Detailed description of patient symptoms
- patient_id (string, required): Unique patient identifier
- image (file, optional): Image of wound/condition (JPEG, PNG)
```

**Response (200 OK):**
```json
{
  "case_id": "64f7a8b2c9e1d2a3b4c5d6e7",
  "patient_id": "PATIENT_001",
  "urgency_level": "HIGH",
  "severity_score": 8.5,
  "assessment": "Possible acute coronary syndrome. Requires immediate emergency evaluation.",
  "recommendations": [
    "Call emergency services (911/100)",
    "Aspirin administration if not contraindicated",
    "ECG monitoring",
    "Cardiac enzyme tests"
  ],
  "confidence": 0.92,
  "created_at": "2025-11-22T21:00:00Z",
  "status": "pending_review"
}
```

**Error Response (400 Bad Request):**
```json
{
  "detail": "Description and patient_id are required"
}
```

**Error Response (500 Internal Server Error):**
```json
{
  "detail": "Failed to process assessment: OPENAI_API_KEY not configured"
}
```

---

### Doctor Review System

#### `GET /triage/pending-reviews`

Retrieve all cases awaiting doctor review.

**Request:**
```bash
curl "http://127.0.0.1:8000/triage/pending-reviews"
```

**Query Parameters:**
```
- limit (integer, optional, default: 20): Number of cases to return
- skip (integer, optional, default: 0): Number of cases to skip (pagination)
```

**Response (200 OK):**
```json
{
  "total": 5,
  "cases": [
    {
      "case_id": "64f7a8b2c9e1d2a3b4c5d6e7",
      "patient_id": "PATIENT_001",
      "description": "Severe chest pain with shortness of breath",
      "urgency_level": "HIGH",
      "severity_score": 8.5,
      "assessment": "Possible acute coronary syndrome...",
      "created_at": "2025-11-22T21:00:00Z",
      "has_image": true
    }
  ]
}
```

---

#### `GET /triage/case/{case_id}`

Retrieve detailed information for a specific case.

**Request:**
```bash
curl "http://127.0.0.1:8000/triage/case/64f7a8b2c9e1d2a3b4c5d6e7"
```

**Path Parameters:**
```
- case_id (string, required): Unique case identifier
```

**Response (200 OK):**
```json
{
  "case_id": "64f7a8b2c9e1d2a3b4c5d6e7",
  "patient_id": "PATIENT_001",
  "description": "Severe chest pain with shortness of breath",
  "urgency_level": "HIGH",
  "severity_score": 8.5,
  "assessment": "Possible acute coronary syndrome. Requires immediate emergency evaluation.",
  "recommendations": [
    "Call emergency services (911/100)",
    "Aspirin administration if not contraindicated",
    "ECG monitoring",
    "Cardiac enzyme tests"
  ],
  "confidence": 0.92,
  "created_at": "2025-11-22T21:00:00Z",
  "status": "pending_review",
  "has_image": true,
  "corrections": []
}
```

**Error Response (404 Not Found):**
```json
{
  "detail": "Case not found"
}
```

---

#### `GET /triage/case/{case_id}/image`

Retrieve the image associated with a case (if available).

**Request:**
```bash
curl "http://127.0.0.1:8000/triage/case/64f7a8b2c9e1d2a3b4c5d6e7/image" \
  -o case_image.jpg
```

**Path Parameters:**
```
- case_id (string, required): Unique case identifier
```

**Response (200 OK):**
```
Binary image data (JPEG/PNG)
Content-Type: image/jpeg
```

**Error Response (404 Not Found):**
```json
{
  "detail": "Image not found for this case"
}
```

---

#### `POST /triage/validate`

Submit doctor review and corrections for an assessment.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/triage/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "64f7a8b2c9e1d2a3b4c5d6e7",
    "doctor_id": "DR_001",
    "corrected_urgency": "CRITICAL",
    "corrected_diagnosis": "Confirmed acute coronary syndrome",
    "feedback": "AI assessment was accurate. Patient referred to cardiology."
  }'
```

**Request Body:**
```json
{
  "case_id": "string (required): Unique case identifier",
  "doctor_id": "string (required): Doctor identifier",
  "corrected_urgency": "string (optional): CRITICAL | HIGH | MEDIUM | LOW",
  "corrected_diagnosis": "string (optional): Doctor's diagnosis",
  "feedback": "string (optional): Detailed feedback for learning"
}
```

**Response (200 OK):**
```json
{
  "case_id": "64f7a8b2c9e1d2a3b4c5d6e7",
  "status": "reviewed",
  "learning_update": "Pattern stored for future reference",
  "ai_accuracy_improved": true,
  "message": "Assessment validated and AI learning updated"
}
```

**Error Response (400 Bad Request):**
```json
{
  "detail": "Case ID is required"
}
```

**Error Response (404 Not Found):**
```json
{
  "detail": "Case not found"
}
```

---

### Analytics

#### `GET /triage/analytics`

Retrieve system analytics and AI performance metrics.

**Request:**
```bash
curl "http://127.0.0.1:8000/triage/analytics"
```

**Response (200 OK):**
```json
{
  "total_assessments": 156,
  "total_reviews": 142,
  "pending_reviews": 5,
  "ai_accuracy": 0.89,
  "average_confidence": 0.87,
  "urgency_distribution": {
    "CRITICAL": 12,
    "HIGH": 34,
    "MEDIUM": 78,
    "LOW": 32
  },
  "learning_patterns_identified": 47,
  "system_uptime_hours": 720
}
```

---

#### `GET /triage/recent-cases`

Retrieve recently processed cases.

**Request:**
```bash
curl "http://127.0.0.1:8000/triage/recent-cases?limit=10&status=all"
```

**Query Parameters:**
```
- limit (integer, optional, default: 10): Number of cases to return
- status (string, optional, default: "all"): Filter by status
  Options: "all", "pending_review", "reviewed"
```

**Response (200 OK):**
```json
{
  "total": 10,
  "cases": [
    {
      "case_id": "64f7a8b2c9e1d2a3b4c5d6e7",
      "patient_id": "PATIENT_001",
      "urgency_level": "HIGH",
      "severity_score": 8.5,
      "status": "reviewed",
      "created_at": "2025-11-22T21:00:00Z"
    }
  ]
}
```

---

### Admin Tools

#### `GET /admin/stats`

Retrieve comprehensive system statistics (admin only).

**Request:**
```bash
curl "http://127.0.0.1:8000/admin/stats" \
  -H "Authorization: Bearer ADMIN_TOKEN"
```

**Response (200 OK):**
```json
{
  "total_cases": 156,
  "total_patients": 98,
  "total_doctors": 12,
  "database_size_mb": 45.2,
  "api_requests_today": 342,
  "error_rate": 0.02,
  "cache_hit_rate": 0.78,
  "average_response_time_ms": 245
}
```

---

#### `POST /admin/clear-cache`

Clear pattern cache and reset learning state (admin only).

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/admin/clear-cache" \
  -H "Authorization: Bearer ADMIN_TOKEN"
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Pattern cache cleared successfully",
  "timestamp": "2025-11-22T21:00:00Z"
}
```

---

#### `POST /admin/reset-system`

Reset entire system to initial state (admin only - use with caution).

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/admin/reset-system" \
  -H "Authorization: Bearer ADMIN_TOKEN"
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "System reset completed",
  "cases_deleted": 156,
  "timestamp": "2025-11-22T21:00:00Z"
}
```

---

#### `POST /admin/recalculate-scores`

Recalculate all severity scores based on current criteria (admin only).

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/admin/recalculate-scores" \
  -H "Authorization: Bearer ADMIN_TOKEN"
```

**Response (200 OK):**
```json
{
  "status": "success",
  "cases_processed": 156,
  "scores_updated": 142,
  "timestamp": "2025-11-22T21:00:00Z"
}
```

---

#### `GET /admin/prompt`

Retrieve the current AI prompt template (admin only).

**Request:**
```bash
curl "http://127.0.0.1:8000/admin/prompt" \
  -H "Authorization: Bearer ADMIN_TOKEN"
```

**Response (200 OK):**
```json
{
  "model": "gpt-4o",
  "system_prompt": "You are an expert medical triage AI assistant...",
  "version": "1.0.0",
  "last_updated": "2025-11-22T21:00:00Z"
}
```

---

#### `GET /admin/pattern-context`

Retrieve stored learning patterns (admin only).

**Request:**
```bash
curl "http://127.0.0.1:8000/admin/pattern-context" \
  -H "Authorization: Bearer ADMIN_TOKEN"
```

**Response (200 OK):**
```json
{
  "total_patterns": 47,
  "patterns": [
    {
      "symptom_cluster": "chest_pain_dyspnea",
      "common_diagnoses": ["ACS", "PE", "Pneumonia"],
      "frequency": 23,
      "average_accuracy": 0.91
    }
  ],
  "last_updated": "2025-11-22T21:00:00Z"
}
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | Your OpenAI API key for GPT-4o |
| `MONGODB_URI` | No | `mongodb://localhost:27017/` | MongoDB connection string |
| `MONGODB_DB` | No | `medical_triage` | Database name |
| `TRIAGE_MODEL` | No | `gpt-4o` | OpenAI model to use |
| `DEBUG` | No | `false` | Enable debug logging |
| `API_PORT` | No | `8000` | API server port |

---

## Example Workflows

### Workflow 1: Patient Assessment

```bash
# 1. Patient submits symptoms
curl -X POST "http://127.0.0.1:8000/triage/assess" \
  -F "description=Persistent headache with fever for 3 days" \
  -F "patient_id=PATIENT_002"

# 2. Get case ID from response
# case_id = "64f7a8b2c9e1d2a3b4c5d6e8"

# 3. Doctor checks pending cases
curl "http://127.0.0.1:8000/triage/pending-reviews"

# 4. Doctor reviews specific case
curl "http://127.0.0.1:8000/triage/case/64f7a8b2c9e1d2a3b4c5d6e8"

# 5. Doctor submits validation
curl -X POST "http://127.0.0.1:8000/triage/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "64f7a8b2c9e1d2a3b4c5d6e8",
    "doctor_id": "DR_002",
    "corrected_urgency": "HIGH",
    "corrected_diagnosis": "Meningitis - requires hospitalization"
  }'
```

### Workflow 2: Analytics Review

```bash
# Check system performance
curl "http://127.0.0.1:8000/triage/analytics"

# View recent cases
curl "http://127.0.0.1:8000/triage/recent-cases?limit=20&status=reviewed"
```

### Workflow 3: Admin Maintenance

```bash
# View system statistics
curl "http://127.0.0.1:8000/admin/stats"

# View learning patterns
curl "http://127.0.0.1:8000/admin/pattern-context"

# Recalculate scores
curl -X POST "http://127.0.0.1:8000/admin/recalculate-scores"
```

---

## Troubleshooting

### Issue: "OPENAI_API_KEY not set"

**Solution:**
1. Create `.env` file from `.env.template`
2. Add your OpenAI API key: `OPENAI_API_KEY=sk-...`
3. Restart the server

```bash
cp .env.template .env
nano .env  # Edit and add your key
uvicorn app.main:app --reload
```

---

### Issue: "MongoDB connection failed"

**Solution:**
1. Ensure MongoDB is running
2. Check connection string in `.env`
3. Verify MongoDB port (default: 27017)

```bash
# Check if MongoDB is running
docker ps | grep mongodb

# Or restart MongoDB
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

---

### Issue: Import errors or missing dependencies

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

### Issue: Slow API responses

**Solution:**
1. Check OpenAI API status
2. Review error logs for bottlenecks
3. Enable debug mode: `DEBUG=true`
4. Check MongoDB performance

```bash
DEBUG=true uvicorn app.main:app --reload
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ^0.104.0 | Web framework |
| `uvicorn` | ^0.24.0 | ASGI server |
| `pymongo` | ^4.5.0 | MongoDB driver |
| `openai` | ^1.0.0 | OpenAI API client |
| `python-multipart` | ^0.0.6 | File upload handling |
| `pydantic` | ^2.0.0 | Data validation |
| `streamlit` | ^1.28.0 | UI framework (optional) |

---

## Security Considerations

- **API Key Protection**: Never commit `.env` files to version control
- **Input Validation**: All inputs are validated using Pydantic models
- **Rate Limiting**: Implement rate limiting for production deployment
- **Authentication**: Add JWT or OAuth2 for production use
- **HTTPS**: Use HTTPS in production
- **Data Privacy**: Ensure HIPAA compliance for healthcare data
- **Audit Logging**: All doctor corrections are logged for accountability

---

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t medical-triage-api .

# Run container
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key \
  -e MONGODB_URI=mongodb://mongo:27017/ \
  --link mongodb:mongo \
  medical-triage-api
```

### Production Checklist

- [ ] Enable authentication and authorization
- [ ] Set up HTTPS/SSL certificates
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerting
- [ ] Enable detailed logging
- [ ] Configure automated backups for MongoDB
- [ ] Set up CI/CD pipeline
- [ ] Conduct security audit
- [ ] Ensure HIPAA compliance

---

## 📞 Support & Contribution

For issues, feature requests, or contributions:

1. Check existing issues on GitHub
2. Create detailed bug reports with reproduction steps
3. Follow code style guidelines
4. Submit pull requests for review

---

## License

This project is for educational and demonstration purposes. For production use, ensure proper security, authentication, and compliance with healthcare regulations (HIPAA, GDPR, etc.).

---

## Medical Disclaimer

**This system is designed as a decision-support tool only.** It should NOT be used as a substitute for professional medical judgment. All assessments must be reviewed and validated by qualified healthcare professionals before clinical decisions are made.

---

**Last Updated:** November 22, 2025  
**Version:** 1.0.0