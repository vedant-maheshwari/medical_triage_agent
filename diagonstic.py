import os
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import traceback

# Load environment variables
load_dotenv()

def mongodb_diagnostic():
    """Comprehensive MongoDB and RAG diagnostic script"""
    print("=" * 80)
    print("🔍 MONGODB & RAG SYSTEM DIAGNOSTIC")
    print("=" * 80)
    
    # 1. Check MongoDB connection
    print("\n1. MongoDB Connection Check")
    try:
        # Get connection string from environment or settings
        mongo_uri = os.getenv("MONGODB_URI") or os.getenv("MONGO_URL") or "mongodb://localhost:27017"
        print(f"Attempting connection to: {mongo_uri.split('@')[-1]}")
        
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        
        # Get database name
        db_name = os.getenv("MONGODB_DB_NAME") or "wound_triage"
        db = client[db_name]
        print(f"✅ Connected to database: {db_name}")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False
    
    # 2. Check required collections
    print("\n2. Collection Check")
    required_collections = ["feedback", "learning_memory", "users", "prompts"]
    existing_collections = db.list_collection_names()
    
    for collection in required_collections:
        if collection in existing_collections:
            count = db[collection].count_documents({})
            print(f"✅ Collection '{collection}' exists ({count} documents)")
        else:
            print(f"❌ Collection '{collection}' missing")
    
    # 3. Check vector search indexes
    print("\n3. Vector Search Index Check")
    try:
        learning_memory = db["learning_memory"]
        
        # Check for Atlas Vector Search indexes
        try:
            indexes = learning_memory.list_search_indexes()
            vector_indexes = [idx for idx in indexes if idx.get('type') == 'vectorSearch']
            
            if vector_indexes:
                print("✅ Vector Search Indexes found:")
                for idx in vector_indexes:
                    print(f"   - {idx['name']} on field: {idx['definition']['mappings']['fields']}")
            else:
                print("⚠️ No Vector Search indexes found. Using fallback brute-force search.")
        except Exception as e:
            print(f"⚠️ Could not check Atlas Vector Search indexes: {e}")
            print("   This is expected if not using MongoDB Atlas or if permissions are limited")
        
        # Check document structure in learning_memory
        sample_doc = learning_memory.find_one({"complaint_embedding": {"$exists": True}})
        if sample_doc:
            print("✅ Found documents with embeddings in learning_memory")
            print(f"   Sample document structure:")
            print(f"   - Case ID: {sample_doc.get('case_id', 'N/A')}")
            print(f"   - Complaint text length: {len(sample_doc.get('complaint_text', ''))}")
            print(f"   - Embedding dimension: {len(sample_doc.get('complaint_embedding', []))}")
            print(f"   - Error types: {sample_doc.get('error_types', [])}")
            print(f"   - Timestamp: {sample_doc.get('timestamp', 'N/A')}")
        else:
            print("⚠️ No documents with embeddings found in learning_memory collection")
            # Check if there are any documents at all
            any_doc = learning_memory.find_one()
            if any_doc:
                print("   Found document without embeddings:")
                print(f"   - Keys: {list(any_doc.keys())}")
            else:
                print("   No documents found in learning_memory collection")
    except Exception as e:
        print(f"❌ Error checking vector indexes: {e}")
        traceback.print_exc()
    
    # 4. Check embedding model
    print("\n4. Embedding Model Check")
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not set in environment")
        else:
            print("✅ OPENAI_API_KEY is set")
            
            # Test embedding generation
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key
            )
            test_embedding = embedding_model.embed_query("Test query for diagnostic")
            print(f"✅ Embedding model working (dimension: {len(test_embedding)})")
            
            # Test similarity calculation
            test_embedding2 = embedding_model.embed_query("Another test query")
            vec1 = np.array(test_embedding)
            vec2 = np.array(test_embedding2)
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            print(f"✅ Similarity calculation working (test similarity: {similarity:.3f})")
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        traceback.print_exc()
    
    # 5. Test RAG retrieval
    print("\n5. RAG Retrieval Test")
    try:
        from app.services import get_relevant_lessons  # Import your function
        
        test_queries = [
            "Patient has a wound on their leg that won't heal",
            "Chest pain when walking upstairs",
            "Headache with blurred vision"
        ]
        
        for query in test_queries:
            print(f"\n   Testing query: '{query}'")
            results = get_relevant_lessons(query, limit=2)
            if results:
                print(f"   ✅ Found {len(results)} relevant lessons")
                for i, result in enumerate(results, 1):
                    similarity = result.get('_similarity', result.get('_relevance_score', 'N/A'))
                    complaint = result.get('complaint_text', 'No complaint text')
                    specialties = f"{result.get('ai_specialty', 'N/A')} → {result.get('doctor_specialty', 'N/A')}"
                    print(f"      {i}. Similarity: {similarity:.3f} | {complaint[:60]}... | {specialties}")
            else:
                print(f"   ❌ No relevant lessons found for query")
    except Exception as e:
        print(f"❌ RAG retrieval test failed: {e}")
        traceback.print_exc()
        print("   Unable to import get_relevant_lessons function - check import path")
    
    # 6. System configuration check
    print("\n6. System Configuration")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Python version: {os.sys.version.split()[0]}")
    print(f"   Environment variables loaded: {'Yes' if load_dotenv() else 'No'}")
    
    # 7. Recent corrections check
    print("\n7. Recent Corrections Analysis")
    try:
        feedback_collection = db["feedback"]
        recent_corrections = list(feedback_collection.find(
            {"feedback_given": True},
            {"timestamp": 1, "ai_specialty": 1, "doctor_specialty": 1, "correction_score": 1}
        ).sort("timestamp", -1).limit(5))
        
        if recent_corrections:
            print(f"✅ Found {len(recent_corrections)} recent corrections:")
            for correction in recent_corrections:
                timestamp = correction.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')
                print(f"   - {timestamp}: {correction.get('ai_specialty')} → {correction.get('doctor_specialty')} (Score: {correction.get('correction_score', 'N/A'):.2f})")
        else:
            print("⚠️ No recent corrections found in feedback collection")
    except Exception as e:
        print(f"❌ Error analyzing recent corrections: {e}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    
    # Summary of potential issues
    issues = []
    
    # Check for common RAG problems from the code analysis
    try:
        from app.services import get_relevant_lessons
        source_code = inspect.getsource(get_relevant_lessons)
        
        if "complaint_embedding_index" in source_code:
            issues.append("⚠️ Code references 'complaint_embedding_index' but this Atlas index may not exist")
        if "$vectorSearch" in source_code:
            issues.append("⚠️ Code uses Atlas Vector Search but your MongoDB may not support it")
        if "fallback" in source_code.lower():
            print("✅ Code has fallback mechanism for vector search")
        else:
            issues.append("⚠️ No fallback mechanism for vector search - will fail if Atlas index missing")
    except:
        issues.append("⚠️ Could not analyze get_relevant_lessons function source code")
    
    if issues:
        print("\n🚨 POTENTIAL ISSUES DETECTED:")
        for issue in issues:
            print(issue)
    else:
        print("\n✅ No obvious configuration issues detected")
    
    return True

# Run the diagnostic
if __name__ == "__main__":
    try:
        import inspect
        mongodb_diagnostic()
    except Exception as e:
        print(f"❌ Critical error running diagnostic: {e}")
        traceback.print_exc()