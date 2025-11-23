import os
from pymongo import MongoClient
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

# Setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGODB_DB", "medical_triage")
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def seed_users():
    print("🌱 Seeding initial users...")
    
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    users_coll = db["users"]
    
    # Clear existing users to avoid hash mismatch
    users_coll.delete_many({"username": {"$in": ["doctor", "admin"]}})
    print("🧹 Cleared existing doctor/admin users")
    
    # Default Doctor
    doctor = {
        "username": "doctor",
        "full_name": "Dr. Aiden",
        "role": "doctor",
        "hashed_password": get_password_hash("doctor123"),
        "disabled": False
    }
    users_coll.insert_one(doctor)
    print("✅ Created user: doctor (password: doctor123)")
        
    # Default Admin
    admin = {
        "username": "admin",
        "full_name": "System Admin",
        "role": "admin",
        "hashed_password": get_password_hash("admin123"),
        "disabled": False
    }
    users_coll.insert_one(admin)
    print("✅ Created user: admin (password: admin123)")

if __name__ == "__main__":
    seed_users()
