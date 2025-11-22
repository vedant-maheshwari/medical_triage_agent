# Medical Triage AI API

A FastAPI-based medical triage system with AI-powered symptom analysis and continuous learning capabilities.

## Features

- 🤖 **AI-Powered Triage**: Analyze patient symptoms using GPT-4o with vision support
- 📸 **Image Analysis**: Upload images of wounds/conditions for visual assessment
- 👨‍⚕️ **Doctor Review System**: Doctors can review and correct AI assessments
- 🧠 **Learning System**: AI learns from doctor corrections using embeddings and pattern analysis
- 📊 **Analytics Dashboard**: Track AI performance and learning progress
- 🔧 **Admin Tools**: System management and maintenance utilities

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Environment

```bash
# Copy template and edit with your values
cp .env.template .env
# Edit .env and add your OPENAI_API_KEY
```

Required in `.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
MONGODB_URI=mongodb://localhost:27017/
```

### 3. Start MongoDB

```bash
# Using Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Or using Homebrew (macOS)
brew services start mongodb-community
```

### 4. Start API Server

```bash
uvicorn app.main:app --reload
```

API will be available at: `http://127.0.0.1:8000`

### 5. (Optional) Start Streamlit UI

```bash
streamlit run streamlit_ui.py
```

UI will be available at: `http://localhost:8501`

## API Documentation

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## Quick Test

```bash
# Health check (no API key needed)
curl http://127.0.0.1:8000/health

# Triage assessment (requires API key)
curl -X POST "http://127.0.0.1:8000/triage/assess" \
  -F "description=Severe chest pain with shortness of breath" \
  -F "patient_id=TEST_001"
```

## Documentation

- **Setup Guide**: See `SETUP.md` for detailed setup instructions
- **cURL Examples**: See `CURL_EXAMPLES.md` for complete API usage examples
- **Quick Test**: See `QUICK_TEST.md` for quick reference

## Project Structure

```
wound_triage_api/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application and endpoints
│   ├── services.py       # Business logic and AI agent
│   ├── models.py         # Pydantic models
│   ├── config.py         # Configuration and settings
│   └── database.py       # MongoDB connection
├── streamlit_ui.py       # Streamlit frontend (calls API)
├── full_app.py          # Original Streamlit app (reference)
├── requirements.txt     # Python dependencies
├── .env.template        # Environment variables template
└── test_api.sh          # Automated test script
```

## API Endpoints

### Patient Portal
- `POST /triage/assess` - Submit symptoms for AI analysis

### Doctor Review
- `GET /triage/pending-reviews` - Get cases awaiting review
- `GET /triage/case/{case_id}` - Get case details
- `GET /triage/case/{case_id}/image` - Get case image
- `POST /triage/validate` - Submit doctor correction

### Analytics
- `GET /triage/analytics` - Get learning analytics
- `GET /triage/recent-cases` - Get recent cases

### Admin
- `GET /admin/stats` - Get admin statistics
- `POST /admin/clear-cache` - Clear pattern cache
- `POST /admin/reset-system` - Reset system
- `POST /admin/recalculate-scores` - Recalculate scores
- `GET /admin/prompt` - Get current prompt
- `GET /admin/pattern-context` - Get pattern context

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key |
| `MONGODB_URI` | No | `mongodb://localhost:27017/` | MongoDB connection |
| `MONGODB_DB` | No | `medical_triage` | Database name |
| `TRIAGE_MODEL` | No | `gpt-4o` | OpenAI model |
| `DEBUG` | No | `false` | Debug mode |

*Required only for AI analysis endpoints

## Troubleshooting

### "OPENAI_API_KEY not set"
1. Create `.env` file from `.env.template`
2. Add your OpenAI API key
3. Restart the server

### "MongoDB connection failed"
1. Ensure MongoDB is running
2. Check `MONGODB_URI` in `.env`

### Import errors
```bash
pip install -r requirements.txt
```

## License

This project is for educational/demonstration purposes.

## Contributing

This is a demonstration project. For production use, ensure proper security, authentication, and compliance with healthcare regulations.
# medical_triage_agent
