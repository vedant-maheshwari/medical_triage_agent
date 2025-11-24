import os
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Connect to MongoDB
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
db_name = os.getenv("MONGODB_DB_NAME", "wound_triage")
client = MongoClient(mongo_uri)
db = client[db_name]

print(f"🔍 Connected to database: {db_name}")
print("=" * 70)

# List all collections
collections = db.list_collection_names()
print(f"📦 Collections: {collections}")
print()

# Inspect 'feedback' collection
print("📄 INSPECTING 'feedback' COLLECTION (last 2 documents):")
feedback = db["feedback"]
for doc in feedback.find().sort("_id", -1).limit(2):
    print(f"  ID: {doc['_id']}")
    print(f"  Patient ID: {doc.get('patient_id', 'N/A')}")
    print(f"  Input Text: {doc.get('input_text', '')[:80]}...")
    print(f"  AI Specialty: {doc.get('ai_specialty')}, Severity: {doc.get('ai_severity')}")
    print(f"  Doctor Specialty: {doc.get('doctor_specialty')}, Severity: {doc.get('doctor_severity')}")
    print(f"  Feedback Given: {doc.get('feedback_given', False)}")
    print(f"  Learned From: {doc.get('learned_from', False)}")
    print(f"  AI Notes Len: {len(doc.get('ai_notes') or '')}, Doctor Notes Len: {len(doc.get('doctor_notes') or '')}")
    print("-" * 50)

print()

# Inspect 'learning_memory' collection
print("🧠 INSPECTING 'learning_memory' COLLECTION:")
if "learning_memory" not in collections:
    print("❌ 'learning_memory' collection DOES NOT EXIST")
else:
    lessons = db["learning_memory"]
    count = lessons.count_documents({})
    print(f"✅ Found {count} documents in 'learning_memory'")
    
    if count > 0:
        print("\nSample document structure (first record):")
        sample = lessons.find_one()
        for key, val in sample.items():
            if key.endswith("_embedding"):
                emb = np.array(val)
                print(f"  {key}: shape={emb.shape if emb.size > 0 else 'INVALID'}, type={type(val).__name__}, first 3 values={list(val[:3]) if len(val) >= 3 else val}")
            elif isinstance(val, str) and len(val) > 100:
                print(f"  {key}: {val[:100]}...")
            else:
                print(f"  {key}: {val}")

        # Check embedding dimension (OpenAI = 1536, HuggingFace = often 384, 768, or 1024)
        if "complaint_embedding" in sample:
            emb_dim = len(sample["complaint_embedding"])
            if emb_dim == 1536:
                print("\n✅ Embeddings appear to be from OpenAI (dim=1536)")
            elif emb_dim in [384, 768, 1024]:
                print(f"\n⚠️ Embeddings appear to be from HuggingFace (dim={emb_dim}) – may cause mismatch with current OpenAI model!")
            else:
                print(f"\n❓ Unknown embedding dimension: {emb_dim}")
    else:
        print("⚠️ 'learning_memory' exists but is EMPTY")

print("\n" + "=" * 70)
print("💡 NEXT STEPS BASED ON FINDINGS:")
print("=" * 70)

if "learning_memory" not in collections or db["learning_memory"].count_documents({}) == 0:
    print("1. 🚨 'learning_memory' is missing or empty → No RAG possible")
    print("2. ➕ You need to migrate corrections from 'feedback' to 'learning_memory'")
    print("3. 📘 Generate OpenAI embeddings for all lessons (not HuggingFace!)")

# Check if any feedback has been given but not learned from
pending_corrections = db["feedback"].count_documents({
    "feedback_given": True,
    "learned_from": {"$ne": True}
})
print(f"4. ⏳ {pending_corrections} corrections in 'feedback' not yet learned from")

# Check environment
print(f"5. 🔑 OPENAI_API_KEY set: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")