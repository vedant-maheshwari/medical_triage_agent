import asyncio
import langchain
# Patch for older langchain versions or mismatch
try:
    langchain.llm_cache = None
except:
    pass

from app.services import MedicalTriageAgent

async def verify_fix():
    print("🔍 Verifying Image Analysis Fix...")
    
    agent = MedicalTriageAgent()
    
    # Invalid image to trigger exception (simulating a failure case)
    dummy_image_b64 = "INVALID_BASE64"
    
    description = "Test wound"
    
    try:
        # This should trigger the exception handler in analyze()
        result = await agent.analyze(description, dummy_image_b64)
        
        print("\n✅ Analysis Result:")
        print(f"Notes: {result.notes}")
        
        if "Model response:" in result.notes or "Analysis error:" in result.notes:
            print("\n✅ VERIFIED: Error details are being captured in notes.")
        else:
            print("\n❌ FAILED: Error details missing from notes.")
            
    except Exception as e:
        print(f"❌ Error during verification: {e}")

if __name__ == "__main__":
    asyncio.run(verify_fix())
