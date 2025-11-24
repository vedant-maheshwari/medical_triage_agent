from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any
import base64
import uvicorn
from datetime import datetime
from bson import ObjectId
import os
from datetime import timedelta
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
    initialize_system,
    Triage,
    assign_case,
    create_user,
    get_user_by_username,
    analyze_triage, # Added
    resolve_case, # Added
    resolve_and_train, # Added for treat & train button
    _construct_system_prompt, # Added for prompt inspector
    get_relevant_lessons, # Added for prompt inspector
    delete_case # Added for admin tools
)
from app.models import (
    TriageRequest, 
    TriageResponse, 
    DoctorValidation, 
    LearningAnalytics,
    CaseDetail,
    CorrectionResponse,
    AdminStats,
    User,
    Token,
    CaseResolution, # Added
    Triage # Added
)
from app.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES, 
    create_access_token, 
    get_current_active_user, 
    get_current_doctor,
    get_current_admin,
    verify_password,
    get_password_hash
)
from fastapi.security import OAuth2PasswordRequestForm
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
# Auth Endpoints
# ============================================================

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    print(f"🔐 LOGIN ATTEMPT: username='{form_data.username}', password_len={len(form_data.password)}")
    user = get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# ============================================================
# Patient Portal Endpoints
# ============================================================

@app.post("/triage/assess", response_model=TriageResponse)
async def assess_symptoms(
    description: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    patient_id: Optional[str] = Form(None),
    language: Optional[str] = Form("en")
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
        
        # Analyze with AI (pass language for multilingual support)
        result = analyze_triage(description, image_b64, language)
        
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
            priority=severity_color.get(result.severity, "Unknown"),
            detected_language=language
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
async def get_pending_reviews_endpoint(current_user: User = Depends(get_current_doctor)):
    """Get all cases awaiting doctor review"""
    try:
        pending_cases = get_pending_reviews()
        
        # Filter for pending or assigned to current user (Clinical Workflow)
        filtered_cases = [
            c for c in pending_cases 
            if c.get("clinical_status") == "pending" or c.get("assigned_to") == current_user.username
        ]
        
        result = []
        result = []
        for case in filtered_cases:
            result.append(CaseDetail(
                case_id=str(case["_id"]),
                timestamp=case["timestamp"],
                patient_id=case["patient_id"],
                input_text=case.get("input_text"),
                ai_specialty=case["ai_specialty"],
                ai_severity=case["ai_severity"],
                ai_notes=case["ai_notes"],
                has_image=case.get("input_image") is not None,
                clinical_status=case.get("clinical_status", "pending"),
                ai_status=case.get("ai_status", "pending_review"),
                assigned_to=case.get("assigned_to")
            ))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/triage/case/{case_id}")
async def get_case_detail(case_id: str, current_user: User = Depends(get_current_doctor)):
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
            "correction_score": case.get("correction_score"),
            "clinical_status": case.get("clinical_status", "pending"),
            "ai_status": case.get("ai_status", "pending_review"),
            "assigned_to": case.get("assigned_to")
        }
        
        return case_dict
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/triage/case/{case_id}/image")
async def get_case_image(case_id: str): # Image can be public or protected? Let's keep public for now for simplicity in patient view
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

@app.post("/triage/attend/{case_id}")
async def attend_case(case_id: str, current_user: User = Depends(get_current_doctor)):
    """Lock a case for the current doctor"""
    try:
        assign_case(case_id, current_user.username)
        return {"status": "success", "message": f"Case {case_id} assigned to {current_user.username}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/triage/resolve")
async def resolve_case_endpoint(resolution: CaseResolution, current_user: User = Depends(get_current_doctor)):
    """Mark case as treated (Clinical Workflow)"""
    try:
        resolve_case(resolution.case_id, resolution.clinical_notes)
        return {"status": "success", "message": "Case marked as treated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/triage/resolve-and-train", response_model=CorrectionResponse)
async def resolve_and_train_endpoint(validation: DoctorValidation, current_user: User = Depends(get_current_doctor)):
    """Mark case as treated AND submit feedback to train AI"""
    try:
        correction_score, changes = resolve_and_train(
            case_id=validation.case_id,
            specialty=validation.correct_specialty,
            severity=validation.correct_severity,
            notes=validation.doctor_notes,
            is_admin=(current_user.role == "admin")
        )
        return CorrectionResponse(validation_score=correction_score, changes_made=changes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/triage/validate", response_model=CorrectionResponse)
async def validate_assessment(validation: DoctorValidation, current_user: User = Depends(get_current_doctor)):
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
        
        is_admin = current_user.role == "admin"
        score, changes = submit_correction(validation.case_id, doctor_result, is_admin=is_admin)
        
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
async def get_analytics(current_user: User = Depends(get_current_admin)):
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
            {}, # Show all cases, not just reviewed ones
            {
                "_id": 1, "patient_id": 1, "ai_specialty": 1, "doctor_specialty": 1,
                "correction_score": 1, "timestamp": 1, "severity_gap": 1, "feedback_given": 1
            }
        ).sort("timestamp", -1).limit(limit)
        
        cases = []
        for doc in recent:
            cases.append({
                "case_id": str(doc["_id"]),
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
async def get_admin_stats(current_user: User = Depends(get_current_admin)):
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

@app.post("/admin/prompt/preview")
async def preview_prompt(request: TriageRequest, current_user: User = Depends(get_current_admin)):
    """Preview the system prompt for a given complaint (Admin Tool)"""
    try:
        # Get complaint text from request
        complaint = request.description or ""
        if not complaint:
            raise HTTPException(status_code=400, detail="No complaint text provided")
        
        # Simulate RAG retrieval
        relevant_lessons = get_relevant_lessons(complaint)
        
        # Convert ObjectId to string for JSON serialization
        for lesson in relevant_lessons:
            if "_id" in lesson:
                lesson["_id"] = str(lesson["_id"])
        
        # Construct prompt
        system_prompt = _construct_system_prompt(relevant_lessons)
        
        return {
            "system_prompt": system_prompt,
            "relevant_lessons": relevant_lessons
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/review/{case_id}")
async def admin_review_case(case_id: str, validation: DoctorValidation, current_user: User = Depends(get_current_admin)):
    """Submit admin review/correction for a case"""
    try:
        doctor_result = Triage(
            specialty=validation.correct_specialty,
            severity=validation.correct_severity,
            notes=validation.doctor_notes or ""
        )
        
        score, changes = submit_correction(case_id, doctor_result, is_admin=True)
        
        return CorrectionResponse(
            status="reviewed",
            case_id=case_id,
            validation_score=score,
            changes=changes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/admin/case/{case_id}")
async def delete_case_endpoint(case_id: str, current_user: User = Depends(get_current_admin)):
    """Delete a case (Admin only)"""
    try:
        success = delete_case(case_id)
        if not success:
            raise HTTPException(status_code=404, detail="Case not found or could not be deleted")
        return {"status": "success", "message": f"Case {case_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/clear-cache")
async def clear_cache_endpoint(current_user: User = Depends(get_current_admin)):
    """Clear pattern cache"""
    try:
        clear_pattern_cache()
        return {"status": "success", "message": "Pattern cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/reset-system")
async def reset_system(
    clear_learning_memory: bool = Query(False, description="Also clear learning memory"),
    current_user: User = Depends(get_current_admin)
):
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
async def recalculate_scores(current_user: User = Depends(get_current_admin)):
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
async def get_prompt(current_user: User = Depends(get_current_admin)):
    """Get current base prompt"""
    try:
        return {
            "prompt": get_current_prompt(),
            "length": len(get_current_prompt())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/pattern-context")
async def get_pattern_context_endpoint(current_user: User = Depends(get_current_admin)):
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
    """Serve the frontend entry point"""
    return FileResponse("frontend/index.html")

# Mount static files (must be after specific routes to avoid conflicts)
app.mount("/", StaticFiles(directory="frontend"), name="static")

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
