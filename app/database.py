from functools import lru_cache
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure
from app.config import settings

@lru_cache
def get_mongo_client() -> MongoClient:
    """Singleton MongoDB client (cached to avoid reconnections)"""
    try:
        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
        client.server_info()  # Test connection
        return client
    except ConnectionFailure:
        raise RuntimeError("MongoDB connection failed. Please ensure MongoDB is running.")

def get_collection(name: str) -> Collection:
    """Return any collection in the database"""
    return get_mongo_client()[settings.db_name][name]

# Convenience accessors
def feedback_collection() -> Collection:
    return get_collection("feedback")

def prompts_collection() -> Collection:
    return get_collection("prompts")

def lessons_collection() -> Collection:
    return get_collection("learning_memory")
