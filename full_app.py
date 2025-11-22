# medical_triage_ai_fixed.py
import streamlit as st
import os
import base64
from datetime import datetime, timedelta  # ADDED timedelta
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import pymongo
from bson import ObjectId
import traceback
import numpy as np

# === LangChain Imports ===
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# === Embedding Imports ===
from sentence_transformers import SentenceTransformer

# === MongoDB Setup ===
MONGODB_URI = "mongodb://localhost:27017/"
DB_NAME = "medical_triage"

# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
SPECIALTY_OPTIONS = [
    "General Practitioner", "Emergency", "Cardiologist", "Orthopedic", 
    "Dermatologist", "Vascular Surgeon", "Plastic Surgeon", "Neurologist", 
    "Gastroenterologist", "Pulmonologist", "Rheumatologist", "Otolaryngologist (ENT)",
    "Ophthalmologist", "Urologist", "Gynecologist", "Pediatrician"
]

# Initialize embedding model
@st.cache_resource
def get_embedding_model():
    """Load sentence transformer model once"""
    try:
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.info("Run: pip install sentence-transformers")
        st.stop()

# ------------------------------------------------------------
# MongoDB Connection
# ------------------------------------------------------------
@st.cache_resource
def get_mongo_client():
    """Get MongoDB client (cached to avoid reconnections)"""
    try:
        client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        return client
    except pymongo.errors.ConnectionFailure:
        st.error("❌ MongoDB connection failed. Please ensure MongoDB is running.")
        st.stop()

def get_collection(collection_name: str):
    """Get a MongoDB collection"""
    client = get_mongo_client()
    db = client[DB_NAME]
    return db[collection_name]

# ------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------
class Triage(BaseModel):
    specialty: str = Field(description="Recommended primary specialist")
    severity: int = Field(gt=0, le=5, description="Severity 1-5")
    notes: str = Field(description="Clinical rationale and any uncertainties")

# ------------------------------------------------------------
# Prompt Management (Pristine Base)
# ------------------------------------------------------------
def _default_prompt() -> str:
    """Immutable base prompt with core medical logic"""
    return """You are Dr. AIDEN, a medical triage specialist. Analyze patient complaints comprehensively.

Task: Return JSON with:
- specialty: Primary specialist from {specialties}
- severity: 1-5 (5=critical/life-threatening)
- notes: Brief clinical rationale, including uncertainties

PRIORITY DECISION RULES (Apply in order):

1. LIFE-THREATENING SYMPTOMS → Emergency (Severity 5)
   - Chest pain + shortness of breath, radiation, sweating
   - Severe trauma, uncontrolled bleeding
   - Sudden neurological deficits (stroke signs)
   - Anaphylaxis, severe allergic reactions

2. CHEST PAIN (without clear life threats) → Cardiologist (Severity 3-4)
   - Pressure, tightness, exertional pain
   - Associated with palpitations or mild breathlessness

3. MUSCULOSKELETAL PAIN → Orthopedic (Severity 2-4)
   - Joint pain, limited mobility, swelling
   - Muscle tears, inability to move limb
   - Severity based on pain level and functional limitation

4. SKIN CONDITIONS → Dermatologist (Severity 1-3)
   - Rashes, itching, lesions, ulcers, burns
   - Severe infections/necrosis → Emergency instead

5. WOUNDS WITH CIRCULATION ISSUES → Vascular Surgeon (Severity 3-4)
   - Deep ulcers, poor circulation signs, diabetic wounds

6. POST-SURGICAL/COSMETIC WOUNDS → Plastic Surgeon (Severity 2-3)

7. NEUROLOGICAL SYMPTOMS → Neurologist (Severity 3-5)
   - Headaches, numbness, tingling, dizziness

8. RESPIRATORY SYMPTOMS → Pulmonologist (Severity 2-4)
   - Persistent cough, breathing difficulty (non-emergent)

9. DIGESTIVE ISSUES → Gastroenterologist (Severity 2-3)
   - Abdominal pain, digestive problems

10. UNCLEAR/AMBIGUOUS → General Practitioner (Severity 2)
    - Multiple vague symptoms
    - Insufficient information for specific routing

SEVERITY GUIDELINES:
- 1: Minor, self-care possible
- 2: Mild, outpatient evaluation needed
- 3: Moderate, specific intervention required
- 4: Severe, urgent specialist attention
- 5: Critical, emergency/immediate care

IMPORTANT:
- If multiple specialties apply, choose the most urgent/primary one
- Acknowledge uncertainties in notes when information is incomplete
- Always err on side of caution for severity

Patient Complaint: {desc}

Return JSON: {format_instructions}"""

def get_current_prompt() -> str:
    """Load the current base prompt (never changes)"""
    return _default_prompt()

# ------------------------------------------------------------
# Learning System with Embeddings
# ------------------------------------------------------------
def store_lesson(case_id: str, complaint: str, ai_result: Triage, doctor_result: Triage):
    """Extract, embed, and store a distillable lesson from doctor correction"""
    lessons = get_collection("learning_memory")
    embedding_model = get_embedding_model()
    
    # Extract error types and missed clues
    errors = []
    clues = []
    
    # 1. Specialty error
    if ai_result.specialty != doctor_result.specialty:
        errors.append("specialty_wrong")
        clues.append(f"For complaints like '{complaint[:60]}...', consider {doctor_result.specialty} due to vascular/neuro/skin involvement")
    
    # 2. Severity miscalibration
    severity_gap = abs(ai_result.severity - doctor_result.severity)
    if severity_gap >= 2:
        errors.append("severity_under_estimated" if doctor_result.severity > ai_result.severity else "severity_over_estimated")
        clues.append(f"Severity should be {doctor_result.severity} not {ai_result.severity} - check for functional loss/red flags")
    
    # 3. Notes inadequacy - find critical terms doctor added
    critical_medical_terms = {
        "ulcer", "ischemia", "necrosis", "red flag", "emergency", "rule out", 
        "concerning", "vascular", "neurovascular", "compromise", "systemic"
    }
    
    ai_terms = set(ai_result.notes.lower().split())
    doc_terms = set(doctor_result.notes.lower().split())
    missing_terms = critical_medical_terms.intersection(doc_terms) - critical_medical_terms.intersection(ai_terms)
    
    if missing_terms:
        errors.append("notes_insufficient")
        clues.append(f"Always include: {', '.join(missing_terms)}")
    
    # Only store if there's something to learn
    if not errors:
        return
    
    # Create lesson document
    lesson_text = f"Complaint: {complaint}\nErrors: {', '.join(errors)}\nLessons: {'; '.join(clues)}"
    embedding = embedding_model.encode(lesson_text).tolist()
    
    lesson_doc = {
        "case_id": case_id,
        "timestamp": datetime.now(),
        "complaint_text": complaint,
        "complaint_embedding": embedding,
        "error_types": errors,
        "lessons_learned": clues,
        "severity_gap": severity_gap,
        "ai_specialty": ai_result.specialty,
        "doctor_specialty": doctor_result.specialty
    }
    
    lessons.insert_one(lesson_doc)
    print(f"📚 Embedded lesson stored: {errors}")

def get_relevant_lessons(complaint: str, limit: int = 3) -> List[Dict]:
    """Semantic search for relevant past mistakes using embeddings"""
    lessons = get_collection("learning_memory")
    embedding_model = get_embedding_model()
    
    if not complaint:
        return []
    
    # Embed the current complaint
    query_embedding = embedding_model.encode(complaint).tolist()
    
    # MongoDB Atlas vector search (if available) or fallback to in-memory
    try:
        # For MongoDB Atlas with vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": "complaint_embedding",
                    "numCandidates": 50,
                    "limit": limit,
                    "index": "complaint_embedding_index"
                }
            }
        ]
        relevant_lessons = list(lessons.aggregate(pipeline))
    except Exception:
        # Fallback: brute-force similarity
        all_lessons = list(lessons.find().sort("timestamp", -1).limit(100))
        relevant_lessons = []
        
        for lesson in all_lessons:
            lesson_embedding = np.array(lesson["complaint_embedding"])
            similarity = np.dot(query_embedding, lesson_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(lesson_embedding)
            )
            if similarity > 0.6:  # Threshold
                relevant_lessons.append((lesson, similarity))
        
        relevant_lessons.sort(key=lambda x: x[1], reverse=True)
        relevant_lessons = [lesson for lesson, _ in relevant_lessons[:limit]]
    
    return relevant_lessons

@st.cache_data(ttl=300)  # 5 minute cache
def get_pattern_context() -> str:
    """Build global pattern context from correction statistics"""
    patterns = get_correction_patterns()
    
    if not patterns:
        return ""
    
    context = "\n\n🌐 GLOBAL CORRECTION PATTERNS (last 7 days):\n"
    
    total_corrections = sum(sum(c.values()) for c in patterns.values() if c != "NOTES_INSUFFICIENCY")
    context += f"Total corrections: {total_corrections}\n\n"
    
    # Top specialty confusions
    for ai_spec, corrections in patterns.items():
        if ai_spec == "NOTES_INSUFFICIENCY":
            continue
        for doc_spec, count in corrections.items():
            if count >= 3:  # Only show patterns with >=3 occurrences
                context += f"- '{ai_spec}' → '{doc_spec}' ({count}x)\n"
    
    # Notes inadequacy trend
    notes_issues = patterns.get("NOTES_INSUFFICIENCY", {}).get("requires_clinical_detail", 0)
    if notes_issues >= 5:
        context += f"- {notes_issues} cases had insufficient notes; add red-flag review\n"
    
    context += "\n"
    return context

# ------------------------------------------------------------
# FIXED FUNCTION: Use datetime.timedelta instead of pd.Timedelta
def get_correction_patterns() -> Dict[str, Dict[str, int]]:
    """Load correction statistics from feedback"""
    feedback = get_collection("feedback")
    
    # Calculate date for last 7 days
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    # Specialty corrections
    pipeline = [
        {"$match": {
            "doctor_specialty": {"$ne": None, "$exists": True},
            "feedback_given": True,
            "timestamp": {"$gte": seven_days_ago},
            "$expr": {"$ne": ["$ai_specialty", "$doctor_specialty"]}
        }},
        {"$group": {
            "_id": {
                "ai_specialty": "$ai_specialty",
                "doctor_specialty": "$doctor_specialty"
            },
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}}
    ]
    
    patterns = {}
    for doc in feedback.aggregate(pipeline):
        ai_spec = doc["_id"]["ai_specialty"]
        doc_spec = doc["_id"]["doctor_specialty"]
        patterns.setdefault(ai_spec, {})[doc_spec] = doc["count"]
    
    # Notes inadequacy pattern - NULL-SAFE VERSION
    notes_issues = feedback.count_documents({
        "feedback_given": True,
        "timestamp": {"$gte": seven_days_ago},
        "doctor_notes": {"$exists": True, "$ne": None},  # NULL CHECK
        "ai_notes": {"$exists": True, "$ne": None},      # NULL CHECK
        "$expr": {"$gt": [
            {"$strLenCP": "$doctor_notes"}, 
            {"$multiply": [{"$strLenCP": "$ai_notes"}, 1.5]}
        ]}
    })
    
    if notes_issues > 0:
        patterns["NOTES_INSUFFICIENCY"] = {"requires_clinical_detail": notes_issues}
    
    return patterns

def clear_pattern_cache():
    """Clear cached pattern context"""
    get_pattern_context.clear()

# ------------------------------------------------------------
# LangChain AI Agent
# ------------------------------------------------------------
class MedicalTriageAgent:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=Triage)
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=500)
    
    def analyze(self, desc: Optional[str], image_b64: Optional[str]) -> Triage:
        """Analyze with layered learning context but pristine base prompt"""
        
        if not desc and not image_b64:
            return Triage(
                specialty="General Practitioner",
                severity=2,
                notes="Insufficient information provided for accurate triage"
            )
        
        try:
            # --- Layered Context Assembly ---
            base_prompt = get_current_prompt()  # Immutable
            
            # Global patterns (what AI gets wrong generally)
            global_context = get_pattern_context()
            
            # Case-specific lessons (similar past mistakes)
            case_context = self._build_case_context(desc or "")
            
            # Final prompt assembly
            full_prompt = base_prompt + global_context + case_context
            
            # Prepare variables
            partial_vars = {
                "specialties": ", ".join(SPECIALTY_OPTIONS),
                "format_instructions": self.parser.get_format_instructions(),
                "desc": desc or "No description provided."
            }
            
            if image_b64:
                # Vision mode
                format_vars = partial_vars.copy()
                formatted_prompt = full_prompt.format(**format_vars)
                
                messages = [
                    SystemMessage(content="You are Dr. AIDEN. Always output JSON."),
                    HumanMessage(content=[
                        {
                            "type": "text",
                            "text": f"{formatted_prompt}\n\nReturn ONLY valid JSON."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ])
                ]
                
                response = self.model.invoke(messages)
            else:
                # Text mode
                prompt = PromptTemplate(
                    template=full_prompt,
                    input_variables=["desc"],
                    partial_variables=partial_vars
                )
                chain = prompt | self.model | self.parser
                response = chain.invoke({"desc": desc or "No description provided"})
            
            return self.parser.parse(response.content)
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            print(f"Full trace: {traceback.format_exc()}")
            st.error(f"Analysis failed: {str(e)}")
            
            return Triage(
                specialty="General Practitioner",
                severity=2,
                notes=f"Analysis error: {str(e)} - requires manual review"
            )
    
    def _build_case_context(self, complaint: str) -> str:
        """Build case-specific learning context from past mistakes"""
        relevant_lessons = get_relevant_lessons(complaint, limit=3)
        
        if not relevant_lessons:
            return ""
        
        context = "\n\n💡 RELEVANT PAST LEARNINGS (avoid these mistakes):\n"
        
        for i, lesson in enumerate(relevant_lessons, 1):
            context += f"\n{i}. Similar case ({lesson['timestamp'].strftime('%Y-%m-%d')}):\n"
            for clue in lesson["lessons_learned"]:
                context += f"   - {clue}\n"
        
        context += "\nApply these lessons to current case.\n"
        return context

# ------------------------------------------------------------
# MongoDB Operations
# ------------------------------------------------------------
def save_case(patient_id: str, text: str, image: Optional[bytes], ai_result: Triage):
    """Save initial AI prediction to MongoDB"""
    feedback = get_collection("feedback")
    
    document = {
        "timestamp": datetime.now(),
        "patient_id": patient_id,
        "input_text": text,
        "input_image": image,
        "ai_specialty": ai_result.specialty,
        "ai_severity": ai_result.severity,
        "ai_notes": ai_result.notes,
        "doctor_specialty": None,
        "doctor_severity": None,
        "doctor_notes": None,
        "correction_score": None,
        "feedback_given": False,
        "learned_from": False
    }
    
    result = feedback.insert_one(document)
    return str(result.inserted_id)

def get_pending_reviews() -> List[Dict]:
    """Get all cases awaiting doctor review"""
    feedback = get_collection("feedback")
    
    cursor = feedback.find(
        {"feedback_given": False},
        {
            "_id": 1, "timestamp": 1, "patient_id": 1, "input_text": 1,
            "ai_specialty": 1, "ai_severity": 1, "ai_notes": 1,
            "input_image": 1
        }
    ).sort("timestamp", -1)
    
    return list(cursor)

def submit_correction(case_id: str, doctor_result: Triage) -> float:
    """Submit correction and trigger learning"""
    feedback = get_collection("feedback")
    
    case = feedback.find_one({"_id": ObjectId(case_id)})
    if not case:
        raise ValueError("Case not found")
    
    changes = []
    correction_score = 1.0
    
    # Calculate correction score
    if doctor_result.specialty != case["ai_specialty"]:
        correction_score -= 0.5
        changes.append(f"Specialty: {case['ai_specialty']} → {doctor_result.specialty}")
    
    severity_gap = abs(doctor_result.severity - case["ai_severity"])
    if severity_gap > 0:
        correction_score -= (0.3 * severity_gap / 4)
        changes.append(f"Severity: {case['ai_severity']} → {doctor_result.severity}")
    
    ai_notes_len = len(case.get("ai_notes", ""))
    doc_notes_len = len(doctor_result.notes)
    
    if doc_notes_len > ai_notes_len * 1.5:
        correction_score -= 0.2
        changes.append("Notes: Added critical clinical details")
    
    if doc_notes_len < 10:
        correction_score -= 0.1
        changes.append("Notes too brief")
    
    correction_score = max(0, correction_score)
    
    # Reconstruct AI result for lesson extraction
    ai_result = Triage(
        specialty=case["ai_specialty"],
        severity=case["ai_severity"],
        notes=case["ai_notes"]
    )
    
    # Store learning lesson
    store_lesson(
        case_id=str(case["_id"]),
        complaint=case["input_text"] or "",
        ai_result=ai_result,
        doctor_result=doctor_result
    )
    
    # Clear caches
    clear_pattern_cache()
    
    # Update feedback record
    feedback.update_one(
        {"_id": ObjectId(case_id)},
        {"$set": {
            "doctor_specialty": doctor_result.specialty,
            "doctor_severity": doctor_result.severity,
            "doctor_notes": doctor_result.notes,
            "correction_score": correction_score,
            "feedback_given": True,
            "learned_from": True,
            "severity_gap": severity_gap
        }}
    )
    
    # Log training update
    print("\n" + "="*60)
    print("🎓 TRAINING UPDATE")
    print("="*60)
    print(f"Case ID: {case_id}")
    print(f"Patient: {case['patient_id']}")
    print(f"Correction Score: {correction_score:.3f}")
    for change in changes:
        print(f"  • {change}")
    print(f"Total Lessons: {get_collection('learning_memory').count_documents({})}")
    print("="*60 + "\n")
    
    return correction_score

def get_learning_analytics() -> Dict:
    """Get comprehensive learning analytics"""
    feedback = get_collection("feedback")
    lessons = get_collection("learning_memory")
    
    # Basic stats
    total_feedback = feedback.count_documents({"feedback_given": True})
    
    avg_pipeline = [
        {"$match": {"feedback_given": True, "correction_score": {"$exists": True}}},
        {"$group": {"_id": None, "avg_score": {"$avg": "$correction_score"}}}
    ]
    avg_result = list(feedback.aggregate(avg_pipeline))
    avg_score = avg_result[0]["avg_score"] if avg_result else 0
    
    # Correction patterns
    pipeline = [
        {"$match": {
            "feedback_given": True,
            "$expr": {"$ne": ["$ai_specialty", "$doctor_specialty"]}
        }},
        {"$group": {
            "_id": {
                "ai_specialty": "$ai_specialty",
                "doctor_specialty": "$doctor_specialty"
            },
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}}
    ]
    
    correction_patterns = {}
    for doc in feedback.aggregate(pipeline):
        ai_spec = doc["_id"]["ai_specialty"]
        doc_spec = doc["_id"]["doctor_specialty"]
        correction_patterns.setdefault(ai_spec, {})[doc_spec] = doc["count"]
    
    # Improvement trend
    trend_pipeline = [
        {"$match": {"feedback_given": True, "correction_score": {"$exists": True}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
            "score": {"$avg": "$correction_score"}
        }},
        {"$sort": {"_id": -1}},
        {"$limit": 30}
    ]
    
    improvement_trend = list(feedback.aggregate(trend_pipeline))
    improvement_list = [{"date": doc["_id"], "score": doc["score"]} for doc in improvement_trend][::-1]
    
    # Lessons learned count
    total_lessons = lessons.count_documents({})
    
    return {
        "total_feedback": total_feedback,
        "avg_correction_score": round(avg_score, 3) if avg_score else 0,
        "correction_patterns": correction_patterns,
        "improvement_trend": improvement_list,
        "total_lessons": total_lessons
    }

def cleanup_corrupted_data():
    """Emergency cleanup"""
    st.warning("🚨 Running emergency cleanup...")
    
    # Reset prompt
    prompts = get_collection("prompts")
    prompts.delete_many({"name": {"$in": ["current_prompt", "prompt_version"]}})
    save_prompt(_default_prompt())
    
    # Clear caches
    clear_pattern_cache()
    
    # Optionally clear learning memory (use with caution)
    if st.checkbox("Also clear learning memory?"):
        get_collection("learning_memory").delete_many({})
        st.warning("Learning memory wiped!")
    
    st.success("✅ System restored to clean state")

def recalculate_all_scores():
    """Recalculate correction scores"""
    feedback = get_collection("feedback")
    result = feedback.find({"feedback_given": True})
    
    updated = 0
    for doc in result:
        if "doctor_specialty" in doc:
            correction_score = 1.0
            if doc["doctor_specialty"] != doc["ai_specialty"]:
                correction_score -= 0.5
            if abs(doc["doctor_severity"] - doc["ai_severity"]) > 0:
                correction_score -= 0.3
            if len(doc.get("doctor_notes", "")) < 10:
                correction_score -= 0.2
            
            correction_score = max(0, correction_score)
            
            feedback.update_one(
                {"_id": doc["_id"]},
                {"$set": {"correction_score": correction_score}}
            )
            updated += 1
    
    return updated

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Medical Triage AI",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Portal",
        ["Patient Portal", "Doctor Review", "Learning Analytics", "Admin Tools"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "🟢 System Status: Active\n"
        "🧠 Learning: Enabled\n"
        "📸 Vision API: Ready"
    )
    
    if page == "Patient Portal":
        patient_portal()
    elif page == "Doctor Review":
        doctor_review()
    elif page == "Learning Analytics":
        learning_analytics()
    elif page == "Admin Tools":
        admin_tools()

def patient_portal():
    """Patient-facing portal"""
    st.title("🩺 Medical Triage AI Portal")
    st.markdown("Describe your symptoms or upload an image for AI assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        text_desc = st.text_area(
            "Symptom Description",
            placeholder="Describe your symptoms: location, severity (1-10), duration, triggers...",
            help="Be specific: 'severe arm pain cannot move' vs 'arm pain'"
        )
    
    with col2:
        uploaded_file = st.file_uploader("Upload Image (optional)", type=["jpg", "jpeg", "png"])
        image_b64 = None
        image_bytes = None
        
        if uploaded_file:
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            
            image_b64 = base64.b64encode(image_bytes).decode()
            uploaded_file.seek(0)
            
            st.image(uploaded_file, caption="Image Preview", width="stretch")
    
    if st.button("🔍 Get AI Assessment", type="primary", use_container_width=True):
        if not text_desc and not image_b64:
            st.error("❌ Please provide at least a description or image.")
            return
        
        with st.spinner("🤖 AI analyzing with learned patterns..."):
            agent = MedicalTriageAgent()
            result = agent.analyze(text_desc, image_b64)
            
            patient_id = f"PAT_{int(datetime.now().timestamp())}"
            save_case(patient_id, text_desc, image_bytes, result)
            
            st.success("✅ Assessment complete! A doctor will review this shortly.")
            
            st.subheader("🩺 AI Assessment Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Recommended Specialist", result.specialty)
            with col2:
                st.metric("Severity Level", f"{result.severity}/5")
            with col3:
                severity_color = {
                    5: "🔴 Critical",
                    4: "🟠 Urgent", 
                    3: "🟡 Moderate",
                    2: "🟢 Mild",
                    1: "⚪ Minor"
                }
                st.metric("Priority", severity_color.get(result.severity, "Unknown"))
            
            st.info(result.notes)
            
            st.markdown("---")
            st.subheader("Recommended Action")
            if result.severity >= 4:
                st.error("🚨 Seek immediate medical attention (Emergency/ER)")
            elif result.severity == 3:
                st.warning("📅 Schedule specialist appointment within 1-2 days")
            elif result.severity == 2:
                st.info("📞 Contact your doctor within the week")
            else:
                st.success("🏠 Monitor symptoms, routine follow-up if needed")
            
            st.caption("💡 This assessment will be reviewed by a doctor for validation.")

def doctor_review():
    """Doctor review interface"""
    st.title("👨‍⚕️ Doctor Review Portal")
    st.markdown("Review AI assessments and provide corrections to train the system")
    
    pending_cases = get_pending_reviews()
    
    if not pending_cases:
        st.info("✅ All assessments have been reviewed!")
        return
    
    st.subheader(f"📋 {len(pending_cases)} cases awaiting review")
    
    case = st.selectbox(
        "Select case to review",
        pending_cases,
        format_func=lambda x: f"Case {str(x['_id'])[-6:]} - {x['patient_id']} ({x['timestamp'].strftime('%Y-%m-%d %H:%M')})"
    )
    
    if case:
        st.markdown("---")
        st.subheader("📋 Case Details")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**👤 Patient Info**")
            st.code(f"ID: {case['patient_id']}\nTime: {case['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            st.markdown("**📝 Patient Complaint**")
            if case.get('input_text'):
                st.info(case['input_text'])
            else:
                st.caption("No text description provided")
        
        if case.get('input_image'):
            st.markdown("**📸 Patient Image**")
            try:
                st.image(case['input_image'], width=300, use_column_width=False)
            except Exception as e:
                st.error(f"Image display error: {e}")
        
        st.markdown("---")
        st.subheader("🤖 AI Assessment")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AI Specialist", case['ai_specialty'])
        with col2:
            st.metric("AI Severity", f"{case['ai_severity']}/5")
        with col3:
            confidence = max(0, 1 - (case['ai_severity'] * 0.1))
            st.metric("AI Confidence", f"{confidence:.1%}")
        
        st.info(case['ai_notes'])
        
        st.markdown("---")
        st.subheader("📝 Doctor Correction")
        st.caption("Adjust fields if the AI assessment is incorrect. Your correction trains the system.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                ai_index = SPECIALTY_OPTIONS.index(case['ai_specialty'])
            except ValueError:
                ai_index = 0
            
            corrected_spec = st.selectbox(
                "Corrected Specialist",
                SPECIALTY_OPTIONS,
                index=ai_index
            )
            
            corrected_severity = st.selectbox(
                "Corrected Severity",
                [1, 2, 3, 4, 5],
                index=case['ai_severity'] - 1
            )
        
        with col2:
            corrected_notes = st.text_area(
                "Corrected Clinical Notes",
                value=case['ai_notes'],
                height=150,
                placeholder="Explain the correction and add clinical details the AI missed..."
            )
        
        if st.button("✅ Submit Correction & Train AI", type="primary", use_container_width=True):
            doctor_result = Triage(
                specialty=corrected_spec,
                severity=corrected_severity,
                notes=corrected_notes
            )
            
            with st.spinner("Training AI on your correction..."):
                score = submit_correction(str(case['_id']), doctor_result)
            
            st.success(f"✅ Correction submitted! Learning score: {score:.3f}")
            st.balloons()
            st.rerun()

def learning_analytics():
    """Learning analytics dashboard"""
    st.title("📊 Learning Analytics Dashboard")
    
    analytics = get_learning_analytics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", analytics["total_feedback"])
    
    with col2:
        st.metric("Avg Accuracy", f"{analytics['avg_correction_score']*100:.1f}%")
    
    with col3:
        st.metric("Lessons Learned", analytics["total_lessons"])
    
    with col4:
        target_met = "✅" if analytics["avg_correction_score"] > 0.8 else "⏳"
        st.metric("Target (80%)", target_met)
    
    st.markdown("---")
    
    # Correction patterns heatmap
    if analytics["correction_patterns"]:
        st.subheader("🎯 Correction Patterns")
        
        pattern_data = []
        for ai_spec, corrections in analytics["correction_patterns"].items():
            for doc_spec, count in corrections.items():
                pattern_data.append({
                    "AI Prediction": ai_spec,
                    "Doctor Correction": doc_spec,
                    "Count": count
                })
        
        if pattern_data:
            df = pd.DataFrame(pattern_data)
            pivot_df = df.pivot(index="AI Prediction", columns="Doctor Correction", values="Count").fillna(0)
            
            fig = px.imshow(
                pivot_df.values,
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale="Reds",
                title="AI vs Doctor Corrections Heatmap",
                labels=dict(color="Number of Corrections")
            )
            fig.update_layout(width=800, height=600)
            st.plotly_chart(fig, width="stretch")
    
    # Performance trend
    if analytics["improvement_trend"]:
        st.subheader("📈 Performance Trend")
        
        dates = [item["date"] for item in analytics["improvement_trend"]]
        scores = [item["score"] for item in analytics["improvement_trend"]]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=scores,
            mode='lines+markers',
            name='Correction Score',
            line=dict(color='green', width=3),
            marker=dict(size=8),
            fill='tonexty'
        ))
        
        fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                     annotation_text="Target Threshold (80%)")
        
        fig.update_layout(
            title="AI Accuracy Over Time (Higher is Better)",
            xaxis_title="Date",
            yaxis_title="Average Correction Score",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified'
        )
        st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    
    # Recent cases table
    st.subheader("📝 Recent Cases")
    feedback = get_collection("feedback")
    recent = feedback.find(
        {"feedback_given": True},
        {
            "_id": 1, "patient_id": 1, "ai_specialty": 1, "doctor_specialty": 1,
            "correction_score": 1, "timestamp": 1, "severity_gap": 1
        }
    ).sort("timestamp", -1).limit(15)
    
    cases = []
    for doc in recent:
        cases.append({
            "Case ID": str(doc["_id"])[-6:],
            "Patient": doc["patient_id"],
            "AI Specialist": doc["ai_specialty"],
            "Correction": doc.get("doctor_specialty", "None"),
            "Score": f"{doc.get('correction_score', 0):.3f}",
            "Gap": doc.get("severity_gap", 0),
            "Time": doc["timestamp"].strftime("%m-%d %H:%M")
        })
    
    if cases:
        st.dataframe(pd.DataFrame(cases), width="stretch", hide_index=True)

def admin_tools():
    """Administrative tools"""
    st.title("🔧 Admin Tools")
    st.warning("⚠️ Use these tools carefully. They affect the entire system.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prompts_coll = get_collection("prompts")
        prompt_versions = prompts_coll.count_documents({"name": "prompt_version"})
        st.metric("Prompt Versions", prompt_versions)
    
    with col2:
        feedback_coll = get_collection("feedback")
        total_cases = feedback_coll.count_documents({})
        reviewed_cases = feedback_coll.count_documents({"feedback_given": True})
        st.metric("Reviews Completed", f"{reviewed_cases}/{total_cases}")
    
    with col3:
        lessons_coll = get_collection("learning_memory")
        total_lessons = lessons_coll.count_documents({})
        st.metric("Total Lessons", total_lessons)
    
    st.markdown("---")
    
    # Emergency Actions
    st.subheader("🚨 Emergency Actions")
    
    if st.button("🧹 Reset System to Clean State", type="secondary"):
        cleanup_corrupted_data()
        st.success("✅ System reset complete!")
        st.rerun()
    
    st.markdown("---")
    
    # Maintenance
    st.subheader("🔍 Data Maintenance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Recalculate All Scores"):
            with st.spinner("Recalculating correction scores..."):
                count = recalculate_all_scores()
            st.success(f"✅ Recalculated {count} scores")
    
    with col2:
        if st.button("🗑️ Clear Pattern Cache"):
            clear_pattern_cache()
            st.success("✅ Pattern cache cleared")
            st.rerun()
    
    st.markdown("---")
    
    # Debug Information
    st.subheader("📝 Debug Information")
    
    tab1, tab2 = st.tabs(["Current Base Prompt", "Pattern Context"])
    
    with tab1:
        st.code(get_current_prompt(), language="text", wrap_lines=True)
    
    with tab2:
        context = get_pattern_context()
        if context:
            st.code(context, language="text")
        else:
            st.info("No patterns learned yet. Review more cases to enable learning.")

# Add these functions in the Prompt Management section

def save_prompt(prompt_text: str):
    """Save prompt to MongoDB with versioning"""
    prompts = get_collection("prompts")
    
    # Version history
    version_doc = {
        "name": "prompt_version",
        "version": datetime.now().timestamp(),
        "prompt": prompt_text,
        "created_at": datetime.now()
    }
    prompts.insert_one(version_doc)
    
    # Update current
    prompts.replace_one(
        {"name": "current_prompt"},
        {"name": "current_prompt", "prompt": prompt_text, "updated_at": datetime.now()},
        upsert=True
    )

def reset_prompt_to_default():
    """Emergency reset - clears corrupted prompts"""
    prompts = get_collection("prompts")
    prompts.delete_many({"name": {"$in": ["current_prompt", "prompt_version"]}})
    save_prompt(_default_prompt())

# ------------------------------------------------------------
# Startup & Initialization
# ------------------------------------------------------------
def initialize_system():
    """One-time initialization on startup"""
    prompts = get_collection("prompts")
    
    # Check if clean prompt exists
    if not prompts.find_one({"name": "current_prompt"}):
        print("🚀 Initializing system with clean prompt...")
        save_prompt(_default_prompt())
    
    # Check for corruption
    current = prompts.find_one({"name": "current_prompt"})
    if current and len(current["prompt"]) > 5000:
        print("⚠️ Detected corrupted prompt, resetting...")
        reset_prompt_to_default()
    
    # Ensure indexes
    feedback = get_collection("feedback")
    feedback.create_index("timestamp")
    feedback.create_index("feedback_given")
    
    lessons = get_collection("learning_memory")
    lessons.create_index("timestamp")
    lessons.create_index("case_id")
    
    print("✅ System initialized and ready")

# ------------------------------------------------------------
# Run App
# ------------------------------------------------------------
if __name__ == "__main__":
    initialize_system()
    main()