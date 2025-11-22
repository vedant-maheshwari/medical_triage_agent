# test_learning_verification.py
import asyncio
from app.services import MedicalTriageAgent, get_relevant_lessons
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

async def verify_learning_pipeline():
    agent = MedicalTriageAgent()
    
    print("="*60)
    print("LEARNING PIPELINE VERIFICATION")
    print("="*60)
    
    # Step 1: Check if lessons exist
    lessons = get_relevant_lessons("diabetic foot wound infection", limit=3)
    print(f"\n1. Lessons found for similar complaint: {len(lessons)}")
    
    if lessons:
        print("   ✅ Lesson retrieval working")
        print(f"   Top relevance: {lessons[0].get('_relevance_score', 0):.3f}")
    else:
        print("   ❌ NO LESSONS RETRIEVED!")
        return
    
    # Step 2: Check prompt building
    print("\n2. Testing prompt building...")
    context = agent._build_case_context("diabetic foot wound infection")
    
    if "💡 RELEVANT PAST LEARNINGS" in context:
        print("   ✅ Context is being added to prompt")
        print(f"   Context length: {len(context)} chars")
    else:
        print("   ❌ NO CONTEXT ADDED TO PROMPT!")
    
    # Step 3: Test actual prediction
    print("\n3. Testing prediction with learning...")
    result = await agent.analyze("diabetic foot ulcer with fever", image_b64=None)
    
    print(f"   Specialty: {result.specialty}")
    print(f"   Severity: {result.severity}")
    
    # Should now mention "emergency" or "systemic infection" if learning worked
    if "emergency" in result.notes.lower() or "systemic" in result.notes.lower():
        print("   ✅ AI is applying learned concepts!")
    else:
        print("   ❌ AI not showing signs of learning in output")

if __name__ == "__main__":
    asyncio.run(verify_learning_pipeline())