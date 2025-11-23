import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # MongoDB
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    db_name: str = os.getenv("MONGODB_DB", "medical_triage")
    
    # OpenAI - check both .env file and environment variable
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    triage_model: str = os.getenv("TRIAGE_MODEL", "gpt-4o")
    
    # Application
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    secret_key: str = os.getenv("SECRET_KEY", "supersecretkey123")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

settings = Settings()

# Set OpenAI API key in environment for langchain
# Check both settings and direct environment variable
api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    settings.openai_api_key = api_key

# Specialty options (matches full_app.py)
SPECIALTY_OPTIONS = [
    "General Practitioner", "Emergency", "Cardiologist", "Orthopedic", 
    "Dermatologist", "Vascular Surgeon", "Plastic Surgeon", "Neurologist", 
    "Gastroenterologist", "Pulmonologist", "Rheumatologist", "Otolaryngologist (ENT)",
    "Ophthalmologist", "Urologist", "Gynecologist", "Pediatrician"
]
