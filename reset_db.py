import os
from pymongo import MongoClient
from dotenv import load_dotenv
from seed_users import seed_users

load_dotenv()

# Setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGODB_DB", "wound_triage")

def reset_db():
    print("⚠️  WARNING: This will DELETE ALL DATA in the database.")
    
    client = MongoClient(MONGODB_URI)
    
    # Potential database names
    target_dbs = ["wound_triage", "medical_triage"]
    env_db = os.getenv("MONGODB_DB")
    if env_db and env_db not in target_dbs:
        target_dbs.append(env_db)
    
    print(f"Target Databases to check: {target_dbs}")
    
    for db_name in target_dbs:
        db = client[db_name]
        collections = ["cases", "learning_memory", "feedback", "users", "prompts"]
        
        found_any = False
        for col_name in collections:
            if col_name in db.list_collection_names():
                db[col_name].drop()
                print(f"🗑️  [{db_name}] Dropped collection: {col_name}")
                found_any = True
        
        if not found_any:
            print(f"💨 [{db_name}] No target collections found.")

    print("\n✨ Database cleared successfully!")
    
    # Re-seed users (will use the DB defined in seed_users.py)
    print("\n🌱 Re-seeding users...")
    seed_users()
    print("\n✅ Database reset complete. You can now start fresh.")

if __name__ == "__main__":
    reset_db()
