import os
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
import traceback

print("🔧 RAG SYSTEM SETUP & REPAIR")
print("=" * 60)

# Load environment variables
load_dotenv()
print(f"✅ Environment variables loaded")

# Get MongoDB connection
mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
db_name = os.getenv('MONGODB_DB_NAME', 'wound_triage')
client = MongoClient(mongo_uri)
db = client[db_name]
print(f"✅ Connected to MongoDB: {db_name}")

# 1. Create missing learning_memory collection
if "learning_memory" not in db.list_collection_names():
    print("\n🔧 Creating 'learning_memory' collection...")
    db.create_collection("learning_memory")
    # Create performance indexes
    db.learning_memory.create_index("timestamp")
    db.learning_memory.create_index("case_id")
    db.learning_memory.create_index("semantic_gap")
    print("✅ Collection created with indexes")
else:
    print("\n✅ 'learning_memory' collection already exists")

# 2. Check OpenAI API key
openai_key = os.getenv('OPENAI_API_KEY')
if not openai_key or "sk-" not in openai_key:
    print("❌ CRITICAL: OPENAI_API_KEY is invalid or missing")
    print("   Please ensure your .env file contains a valid OpenAI API key")
    exit(1)
print("✅ OpenAI API key validated")

# 3. Add bootstrap lessons to kickstart RAG
print("\n🌱 ADDING BOOTSTRAP LESSONS...")
from langchain_openai import OpenAIEmbeddings

try:
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_key
    )
    print("✅ Embedding model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load embedding model: {e}")
    exit(1)

# Bootstrap lessons for wound triage
bootstrap_lessons = [
    {
        "complaint": "Patient has a wound on their leg that won't heal after 3 weeks",
        "lesson": "Chronic non-healing wounds require Vascular Surgeon evaluation, not General Practitioner. Duration is critical factor - wounds lasting >2 weeks need vascular assessment for underlying circulation issues.",
        "ai_specialty": "General Practitioner",
        "doctor_specialty": "Vascular Surgeon",
        "ai_severity": 2,
        "doctor_severity": 4
    },
    {
        "complaint": "Deep wound on foot with redness spreading around it",
        "lesson": "Spreading redness around wounds indicates cellulitis/infection requiring Emergency evaluation. Never downgrade severity when infection signs are present.",
        "ai_specialty": "General Practitioner",
        "doctor_specialty": "Emergency",
        "ai_severity": 2,
        "doctor_severity": 5
    },
    {
        "complaint": "Surgical wound from 2 days ago that's opening up",
        "lesson": "Post-surgical wounds that are dehiscing (opening) require immediate Plastic Surgeon evaluation. Time sensitivity is critical - wounds within 7 days post-op need urgent specialist attention.",
        "ai_specialty": "General Practitioner",
        "doctor_specialty": "Plastic Surgeon",
        "ai_severity": 3,
        "doctor_severity": 4
    },
    {
        "complaint": "Patient has chest pain that hurts when I press on my ribs after lifting weights",
        "lesson": "Musculoskeletal chest pain from injury requires Orthopedic evaluation, not Emergency or Cardiologist. Pain that's reproducible with palpation is typically musculoskeletal in origin.",
        "ai_specialty": "Emergency",
        "doctor_specialty": "Orthopedic",
        "ai_severity": 5,
        "doctor_severity": 3
    },
    {
        "complaint": "Headache with blurred vision and numbness in left arm",
        "lesson": "Neurological symptoms including headache with visual changes and numbness require Neurologist evaluation. These could indicate serious neurological conditions requiring specialist assessment.",
        "ai_specialty": "General Practitioner",
        "doctor_specialty": "Neurologist",
        "ai_severity": 2,
        "doctor_severity": 4
    }
]

lessons = db.learning_memory
added_count = 0

for lesson in bootstrap_lessons:
    try:
        # Generate proper embeddings
        complaint_embedding = embedding_model.embed_query(lesson["complaint"])
        lesson_embedding = embedding_model.embed_query(lesson["lesson"])
        
        # Create lesson document
        lesson_doc = {
            "case_id": f"bootstrap_{hash(lesson['complaint'])}",
            "timestamp": datetime.now(),
            "complaint_text": lesson["complaint"],
            "complaint_embedding": complaint_embedding,
            "lesson_embedding": lesson_embedding,
            "error_types": ["specialty_mismatch", "severity_miscalibration"],
            "lessons_learned": [lesson["lesson"]],
            "reasoning_patterns": [
                "Duration of symptoms is critical for wound assessment",
                "Infection signs require immediate escalation",
                "Post-surgical complications need specialist attention within 7 days"
            ],
            "clinical_insights": [
                "Wounds lasting >2 weeks need vascular assessment",
                "Spreading redness indicates cellulitis",
                "Dehiscing wounds require immediate surgical consultation"
            ],
            "ai_specialty": lesson["ai_specialty"],
            "doctor_specialty": lesson["doctor_specialty"],
            "ai_severity": lesson["ai_severity"],
            "doctor_severity": lesson["doctor_severity"],
            "ai_notes": "Initial assessment based on limited information",
            "doctor_notes": lesson["lesson"],
            "semantic_gap": 0.7,
            "severity_gap": abs(lesson["ai_severity"] - lesson["doctor_severity"])
        }
        
        lessons.insert_one(lesson_doc)
        print(f"✅ Added bootstrap lesson: {lesson['complaint'][:50]}...")
        added_count += 1
        
    except Exception as e:
        print(f"⚠️ Error adding bootstrap lesson: {e}")
        traceback.print_exc()

print(f"\n🎉 SUCCESSFULLY ADDED {added_count} BOOTSTRAP LESSONS")

# 4. Verify setup
print("\n" + "=" * 60)
print("✅ SETUP COMPLETE - SYSTEM STATUS")
print("=" * 60)
print(f"learning_memory collection: {lessons.count_documents({})} documents")
print(f"feedback collection: {db.feedback.count_documents({})} documents")
print(f"Pending cases for review: {db.feedback.count_documents({'feedback_given': False})}")

print("\n🚀 NEXT STEPS:")
print("1. Replace your services.py with the fixed version below")
print("2. Restart your application")
print("3. Submit some test cases and provide feedback to see RAG in action")