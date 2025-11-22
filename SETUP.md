# Setup Guide - Medical Triage API

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=medical_triage
TRIAGE_MODEL=gpt-4o
DEBUG=false
```

**Get your OpenAI API key:**
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key and paste it in your `.env` file

### 3. Start MongoDB

**Option A: Using Homebrew (macOS)**
```bash
brew services start mongodb-community
```

**Option B: Using Docker**
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

**Option C: Using MongoDB Atlas (Cloud)**
- Sign up at https://www.mongodb.com/cloud/atlas
- Create a free cluster
- Get connection string and update `MONGODB_URI` in `.env`

### 4. Start the API Server

```bash
uvicorn app.main:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

### 5. (Optional) Start Streamlit UI

In a new terminal:

```bash
streamlit run streamlit_ui.py
```

The UI will be available at: `http://localhost:8501`

## Verify Setup

### Test Health Endpoint (No API key needed)
```bash
curl http://127.0.0.1:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "mongodb": "connected",
  "timestamp": "2024-..."
}
```

### Test Triage Assessment (Requires API key)
```bash
curl -X POST "http://127.0.0.1:8000/triage/assess" \
  -F "description=Severe chest pain with shortness of breath" \
  -F "patient_id=TEST_001"
```

## Troubleshooting

### Error: "OPENAI_API_KEY not set"

**Solution:**
1. Make sure `.env` file exists in project root
2. Check that `OPENAI_API_KEY` is set in `.env`
3. Restart the API server after creating/updating `.env`
4. Or set environment variable directly:
   ```bash
   export OPENAI_API_KEY=your_key_here
   uvicorn app.main:app --reload
   ```

### Error: "MongoDB connection failed"

**Solution:**
1. Make sure MongoDB is running:
   ```bash
   # Check if running
   brew services list | grep mongodb
   # Or
   docker ps | grep mongo
   ```
2. Verify `MONGODB_URI` in `.env` is correct
3. For Docker: Make sure container is running: `docker start mongodb`

### Error: "cannot import name '_get_object_size' from 'bson'"

**Solution:**
```bash
pip uninstall -y bson
pip install --upgrade pymongo
```

### Error: "ModuleNotFoundError: No module named 'pydantic_settings'"

**Solution:**
```bash
pip install pydantic-settings
```

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key for AI analysis |
| `MONGODB_URI` | No | `mongodb://localhost:27017/` | MongoDB connection string |
| `MONGODB_DB` | No | `medical_triage` | Database name |
| `TRIAGE_MODEL` | No | `gpt-4o` | OpenAI model to use |
| `DEBUG` | No | `false` | Enable debug mode |

*Required only for AI analysis endpoints. Other endpoints work without it.

## Next Steps

1. **Test the API**: Run `./test_api.sh`
2. **View API Docs**: Visit http://127.0.0.1:8000/docs
3. **Use Streamlit UI**: Run `streamlit run streamlit_ui.py`
4. **Read Examples**: See `CURL_EXAMPLES.md` for detailed API usage

## Production Deployment

For production:
1. Set `DEBUG=false` in `.env`
2. Use environment variables instead of `.env` file
3. Use a proper MongoDB instance (not localhost)
4. Set up proper CORS origins
5. Use a process manager like `systemd` or `supervisor`
6. Consider using `gunicorn` with `uvicorn` workers

