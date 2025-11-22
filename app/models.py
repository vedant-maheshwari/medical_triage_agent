from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

# Core Triage Model (matches full_app.py)
class Triage(BaseModel):
    specialty: str = Field(description="Recommended primary specialist")
    severity: int = Field(gt=0, le=5, description="Severity 1-5")
    notes: str = Field(description="Clinical rationale and any uncertainties")

# API Request/Response Models
class TriageRequest(BaseModel):
    description: Optional[str] = Field(None, min_length=1, max_length=1000)
    patient_id: Optional[str] = Field(None, description="Optional patient ID")
    
    @validator('description')
    def validate_description(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip() if v else None

class TriageResponse(BaseModel):
    assessment_id: str
    specialty: str
    severity: int
    notes: str
    confidence: float
    urgency_minutes: int
    recommended_action: str
    priority: str

class DoctorValidation(BaseModel):
    case_id: str
    correct_specialty: str
    correct_severity: int
    doctor_notes: Optional[str] = None
    
    @validator('correct_severity')
    def validate_severity(cls, v):
        if not 1 <= v <= 5:
            raise ValueError("Severity must be between 1-5")
        return v

class CaseDetail(BaseModel):
    case_id: str
    timestamp: datetime
    patient_id: str
    input_text: Optional[str]
    ai_specialty: str
    ai_severity: int
    ai_notes: str
    has_image: bool

class LearningAnalytics(BaseModel):
    total_feedback: int
    avg_correction_score: float
    correction_patterns: Dict[str, Dict[str, int]]
    improvement_trend: List[Dict[str, Any]]
    total_lessons: int

class CorrectionResponse(BaseModel):
    status: str
    case_id: str
    validation_score: float
    changes: List[str]

class AdminStats(BaseModel):
    prompt_versions: int
    total_cases: int
    reviewed_cases: int
    total_lessons: int
