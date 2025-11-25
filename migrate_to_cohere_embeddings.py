#!/usr/bin/env python3
"""
migrate_to_cohere_embeddings.py

One-time migration script to re-embed all existing lessons with Cohere's multilingual model.
Run this after switching to Cohere embeddings.
"""

import sys
sys.path.append('.')

from app.services import get_collection, get_embedding_model
from tqdm import tqdm

def migrate_embeddings():
    """Re-embed all lessons with Cohere multilingual embeddings"""
    
    print("=" * 60)
    print("COHERE MULTILINGUAL EMBEDDING MIGRATION")
    print("=" * 60)
    
    lessons = get_collection("learning_memory")
    embedding_model = get_embedding_model()
    
    # Get all lessons
    all_lessons = list(lessons.find({}))
    total = len(all_lessons)
    
    if total == 0:
        print("✅ No lessons found. Nothing to migrate.")
        return
    
    print(f"\n📚 Found {total} lessons to re-embed with Cohere multilingual model...")
    print("This will enable semantic matching across Hindi, Hinglish, and English!\n")
    
    migrated = 0
    failed = 0
    
    for lesson in tqdm(all_lessons, desc="Re-embedding lessons"):
        try:
            complaint = lesson.get('complaint_text', '')
            if not complaint:
                continue
            
            # Generate new multilingual embedding
            new_embedding = embedding_model.embed_query(complaint)
            
            # Update in database
            lessons.update_one(
                {"_id": lesson["_id"]},
                {"$set": {"complaint_embedding": new_embedding}}
            )
            migrated += 1
            
        except Exception as e:
            print(f"\n⚠️  Failed to migrate lesson {lesson.get('_id')}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Migration complete!")
    print(f"   Migrated: {migrated}/{total}")
    if failed > 0:
        print(f"   Failed: {failed}/{total}")
    print("=" * 60)
    print("\n🎉 Your RAG system now supports true multilingual semantic search!")
    print("   - Hindi (Devanagari): मेरे हाथ में दर्द है")
    print("   - Hinglish: mere haath me dard hai")
    print("   - English: pain in my hand")
    print("   All are semantically matched! No translation needed!\n")

if __name__ == "__main__":
    migrate_embeddings()
