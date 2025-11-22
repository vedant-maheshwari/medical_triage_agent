from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List, Dict, Any
import base64
import uvicorn
from datetime import datetime
from bson import ObjectId
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from app.services import (
    MedicalTriageAgent, 
    get_mongo_client, 
    get_collection,
    save_case,
    get_pending_reviews,
    submit_correction,
    get_learning_analytics,
    clear_pattern_cache,
    cleanup_corrupted_data,
    recalculate_all_scores,
    get_current_prompt,
    get_pattern_context,
    initialize_system,
    Triage
)
from app.models import (
    TriageRequest, 
    TriageResponse, 
    DoctorValidation, 
    LearningAnalytics,
    CaseDetail,
    CorrectionResponse,
    AdminStats
)
from app.config import settings, SPECIALTY_OPTIONS

app = FastAPI(
    title="Medical Triage AI API",
    description="AI-powered medical symptom analysis and triage with learning capabilities",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent lazily (singleton pattern)
_agent = None

def get_agent():
    """Get or create the medical triage agent (lazy initialization)"""
    global _agent
    if _agent is None:
        _agent = MedicalTriageAgent()
    return _agent

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    try:
        initialize_system()
    except Exception as e:
        print(f"⚠️ Startup initialization warning: {e}")

# ============================================================
# Patient Portal Endpoints
# ============================================================

@app.post("/triage/assess", response_model=TriageResponse)
async def assess_symptoms(
    description: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    patient_id: Optional[str] = Form(None)
):
    """
    Analyze patient symptoms and return triage recommendation.
    Accepts text description and/or image.
    """
    try:
        if not description and not image:
            raise HTTPException(
                status_code=400, 
                detail="Please provide at least a description or image"
            )
        
        image_b64 = None
        image_bytes = None
        
        if image:
            image_bytes = await image.read()
            image_b64 = base64.b64encode(image_bytes).decode()
        
        # Analyze with AI
        agent = get_agent()
        result = await agent.analyze(description, image_b64)
        
        # Generate patient ID if not provided
        if not patient_id:
            patient_id = f"PAT_{int(datetime.now().timestamp())}"
        
        # Save case to database
        case_id = save_case(patient_id, description or "", image_bytes, result)
        
        # Determine recommended action and priority
        severity_color = {
            5: "🔴 Critical",
            4: "🟠 Urgent", 
            3: "🟡 Moderate",
            2: "🟢 Mild",
            1: "⚪ Minor"
        }
        
        if result.severity >= 4:
            recommended_action = "🚨 Seek immediate medical attention (Emergency/ER)"
        elif result.severity == 3:
            recommended_action = "📅 Schedule specialist appointment within 1-2 days"
        elif result.severity == 2:
            recommended_action = "📞 Contact your doctor within the week"
        else:
            recommended_action = "🏠 Monitor symptoms, routine follow-up if needed"
        
        urgency_minutes = {
            5: 15,
            4: 120,
            3: 1440,
            2: 10080,
            1: 43200
        }.get(result.severity, 10080)
        
        return TriageResponse(
            assessment_id=case_id,
            specialty=result.specialty,
            severity=result.severity,
            notes=result.notes,
            confidence=0.8,  # Can be enhanced later
            urgency_minutes=urgency_minutes,
            recommended_action=recommended_action,
            priority=severity_color.get(result.severity, "Unknown")
        )
    except HTTPException:
        raise
    except ValueError as e:
        # Handle missing API key gracefully
        if "OPENAI_API_KEY" in str(e):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "OpenAI API key not configured",
                    "message": str(e),
                    "help": "Please set OPENAI_API_KEY in .env file or environment variable"
                }
            )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ============================================================
# Doctor Review Endpoints
# ============================================================

@app.get("/triage/pending-reviews", response_model=List[CaseDetail])
async def get_pending_reviews_endpoint():
    """Get all cases awaiting doctor review"""
    try:
        pending_cases = get_pending_reviews()
        
        result = []
        for case in pending_cases:
            result.append(CaseDetail(
                case_id=str(case["_id"]),
                timestamp=case["timestamp"],
                patient_id=case["patient_id"],
                input_text=case.get("input_text"),
                ai_specialty=case["ai_specialty"],
                ai_severity=case["ai_severity"],
                ai_notes=case["ai_notes"],
                has_image=case.get("input_image") is not None
            ))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/triage/case/{case_id}")
async def get_case_detail(case_id: str):
    """Get detailed information about a specific case"""
    try:
        feedback = get_collection("feedback")
        case = feedback.find_one({"_id": ObjectId(case_id)})
        
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Convert ObjectId and bytes for JSON serialization
        case_dict = {
            "case_id": str(case["_id"]),
            "timestamp": case["timestamp"],
            "patient_id": case["patient_id"],
            "input_text": case.get("input_text"),
            "ai_specialty": case["ai_specialty"],
            "ai_severity": case["ai_severity"],
            "ai_notes": case["ai_notes"],
            "has_image": case.get("input_image") is not None,
            "doctor_specialty": case.get("doctor_specialty"),
            "doctor_severity": case.get("doctor_severity"),
            "doctor_notes": case.get("doctor_notes"),
            "feedback_given": case.get("feedback_given", False),
            "correction_score": case.get("correction_score")
        }
        
        return case_dict
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/triage/case/{case_id}/image")
async def get_case_image(case_id: str):
    """Get the image associated with a case"""
    try:
        feedback = get_collection("feedback")
        case = feedback.find_one(
            {"_id": ObjectId(case_id)},
            {"input_image": 1}
        )
        
        if not case or not case.get("input_image"):
            raise HTTPException(status_code=404, detail="Image not found")
        
        from fastapi.responses import Response
        return Response(
            content=case["input_image"],
            media_type="image/jpeg"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/triage/validate", response_model=CorrectionResponse)
async def validate_assessment(validation: DoctorValidation):
    """
    Submit doctor's correction and calculate learning score.
    This trains the AI system.
    """
    try:
        doctor_result = Triage(
            specialty=validation.correct_specialty,
            severity=validation.correct_severity,
            notes=validation.doctor_notes or ""
        )
        
        score, changes = submit_correction(validation.case_id, doctor_result)
        
        return CorrectionResponse(
            status="validated",
            case_id=validation.case_id,
            validation_score=score,
            changes=changes
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Learning Analytics Endpoints
# ============================================================

@app.get("/triage/analytics", response_model=LearningAnalytics)
async def get_analytics():
    """Get comprehensive AI learning analytics"""
    try:
        analytics = get_learning_analytics()
        return LearningAnalytics(**analytics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/triage/recent-cases")
async def get_recent_cases(limit: int = 15):
    """Get recent reviewed cases"""
    try:
        feedback = get_collection("feedback")
        recent = feedback.find(
            {"feedback_given": True},
            {
                "_id": 1, "patient_id": 1, "ai_specialty": 1, "doctor_specialty": 1,
                "correction_score": 1, "timestamp": 1, "severity_gap": 1
            }
        ).sort("timestamp", -1).limit(limit)
        
        cases = []
        for doc in recent:
            cases.append({
                "case_id": str(doc["_id"])[-6:],
                "patient_id": doc["patient_id"],
                "ai_specialty": doc["ai_specialty"],
                "doctor_specialty": doc.get("doctor_specialty", "None"),
                "correction_score": doc.get("correction_score", 0),
                "severity_gap": doc.get("severity_gap", 0),
                "timestamp": doc["timestamp"].isoformat()
            })
        
        return cases
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Admin Tools Endpoints
# ============================================================

@app.get("/admin/stats", response_model=AdminStats)
async def get_admin_stats():
    """Get administrative statistics"""
    try:
        prompts_coll = get_collection("prompts")
        prompt_versions = prompts_coll.count_documents({"name": "prompt_version"})
        
        feedback_coll = get_collection("feedback")
        total_cases = feedback_coll.count_documents({})
        reviewed_cases = feedback_coll.count_documents({"feedback_given": True})
        
        lessons_coll = get_collection("learning_memory")
        total_lessons = lessons_coll.count_documents({})
        
        return AdminStats(
            prompt_versions=prompt_versions,
            total_cases=total_cases,
            reviewed_cases=reviewed_cases,
            total_lessons=total_lessons
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/clear-cache")
async def clear_cache_endpoint():
    """Clear pattern cache"""
    try:
        clear_pattern_cache()
        return {"status": "success", "message": "Pattern cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/reset-system")
async def reset_system(clear_learning_memory: bool = Query(False, description="Also clear learning memory")):
    """
    Reset system to clean state (Emergency action).
    Use with caution!
    Accepts query parameter: clear_learning_memory (bool)
    """
    try:
        cleanup_corrupted_data(clear_learning_memory=clear_learning_memory)
        return {
            "status": "success",
            "message": "System reset complete",
            "learning_memory_cleared": clear_learning_memory
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/recalculate-scores")
async def recalculate_scores():
    """Recalculate all correction scores"""
    try:
        count = recalculate_all_scores()
        return {
            "status": "success",
            "message": f"Recalculated {count} scores",
            "updated_count": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/prompt")
async def get_prompt():
    """Get current base prompt"""
    try:
        return {
            "prompt": get_current_prompt(),
            "length": len(get_current_prompt())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/pattern-context")
async def get_pattern_context_endpoint():
    """Get current pattern context"""
    try:
        context = get_pattern_context()
        return {
            "context": context if context else "No patterns learned yet",
            "has_patterns": bool(context)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/specialties")
async def get_specialties():
    """Get list of available specialties"""
    return {"specialties": SPECIALTY_OPTIONS}

# ============================================================
# Health & Utility Endpoints
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        client = get_mongo_client()
        client.server_info()
        return {
            "status": "healthy",
            "mongodb": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "mongodb": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Medical Triage AI API",
        "version": "1.0.0",
        "description": "AI-powered medical symptom analysis and triage",
        "endpoints": {
            "patient": "/triage/assess",
            "doctor": "/triage/pending-reviews",
            "analytics": "/triage/analytics",
            "admin": "/admin/stats",
            "health": "/health"
        },
        "docs": "/docs"
    }

# ============================================================
# Run App
# ============================================================

if __name__ == "__main__":
    # Run from project root with: uvicorn app.main:app --reload
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )
