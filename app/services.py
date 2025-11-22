# services.py - All business logic from full_app.py converted to FastAPI
import os
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pymongo
from bson import ObjectId
import traceback
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# === LangChain Imports ===
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# === Embedding Imports ===
from sentence_transformers import SentenceTransformer

# === Local Imports ===
from app.database import get_collection, get_mongo_client
from app.config import settings, SPECIALTY_OPTIONS

load_dotenv()

# Initialize embedding model (singleton)
_embedding_model = None

def get_embedding_model():
    """Load sentence transformer model once (singleton)"""
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}. Run: pip install sentence-transformers")
    return _embedding_model

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
def extract_semantic_insights(ai_notes: str, doctor_notes: str, embedding_model) -> Dict[str, Any]:
    """
    Extract semantic insights from doctor notes using embedding-based comparison.
    Understands meaning, not just keywords.
    """
    insights = {
        "missing_concepts": [],
        "reasoning_patterns": [],
        "clinical_priorities": [],
        "semantic_gap": 0.0
    }
    
    if not doctor_notes or len(doctor_notes.strip()) < 10:
        return insights
    
    # Embed both notes for semantic comparison
    ai_embedding = np.array(embedding_model.encode(ai_notes))
    doc_embedding = np.array(embedding_model.encode(doctor_notes))
    
    # Calculate semantic similarity
    similarity = np.dot(ai_embedding, doc_embedding) / (
        np.linalg.norm(ai_embedding) * np.linalg.norm(doc_embedding)
    )
    insights["semantic_gap"] = 1.0 - similarity  # Gap = how different they are
    
    # Split into sentences for detailed analysis
    doc_sentences = [s.strip() for s in doctor_notes.split('.') if len(s.strip()) > 15]
    ai_sentences = [s.strip() for s in ai_notes.split('.') if len(s.strip()) > 15]
    
    # Find semantically unique doctor insights
    if doc_sentences:
        for doc_sent in doc_sentences[:5]:  # Analyze top 5 sentences
            doc_sent_emb = np.array(embedding_model.encode(doc_sent))
            
            # Check if this sentence is semantically similar to any AI sentence
            is_unique = True
            max_ai_similarity = 0.0
            
            for ai_sent in ai_sentences:
                ai_sent_emb = np.array(embedding_model.encode(ai_sent))
                sent_similarity = np.dot(doc_sent_emb, ai_sent_emb) / (
                    np.linalg.norm(doc_sent_emb) * np.linalg.norm(ai_sent_emb)
                )
                max_ai_similarity = max(max_ai_similarity, sent_similarity)
                if sent_similarity > 0.75:  # Very similar semantically
                    is_unique = False
                    break
            
            # If sentence is unique (semantically different), extract insights
            if is_unique and max_ai_similarity < 0.6:
                # Extract key medical concepts and reasoning
                # Look for patterns: condition + consequence + action
                lower_sent = doc_sent.lower()
                
                # Extract clinical priorities (urgency indicators)
                urgency_indicators = ["immediate", "urgent", "critical", "emergent", "emergent", 
                                     "life-threatening", "severe", "acute", "requires", "must"]
                if any(indicator in lower_sent for indicator in urgency_indicators):
                    insights["clinical_priorities"].append(doc_sent[:100])  # First 100 chars
                
                # Extract reasoning patterns (if-then, because, due to, etc.)
                reasoning_keywords = ["because", "due to", "indicates", "suggests", "consistent with",
                                    "requires", "necessitates", "warrants", "indicates need for"]
                if any(keyword in lower_sent for keyword in reasoning_keywords):
                    insights["reasoning_patterns"].append(doc_sent[:120])
                
                # Extract missing concepts (medical terms, conditions, procedures)
                # Filter out common words, keep medical terminology
                words = doc_sent.split()
                medical_concepts = [w.strip('.,!?;:') for w in words 
                                  if len(w) > 5 and w.lower() not in 
                                  ['patient', 'should', 'needs', 'requires', 'consider', 
                                   'assess', 'evaluate', 'indicates', 'suggests']]
                if medical_concepts:
                    insights["missing_concepts"].extend(medical_concepts[:3])  # Top 3 concepts
    
    return insights

def store_lesson(case_id: str, complaint: str, ai_result: Triage, doctor_result: Triage):
    """
    Enhanced lesson extraction with semantic understanding and reasoning pattern learning.
    Uses embeddings to understand meaning, not just keywords.
    """
    lessons = get_collection("learning_memory")
    embedding_model = get_embedding_model()
    
    # Extract error types and missed clues
    errors = []
    clues = []
    reasoning_patterns = []
    clinical_insights = []
    
    # 1. Specialty error with context
    if ai_result.specialty != doctor_result.specialty:
        errors.append("specialty_wrong")
        # Extract why doctor chose different specialty from their notes
        specialty_reasoning = ""
        if doctor_result.notes:
            # Look for specialty-related reasoning in doctor notes
            specialty_keywords = {
                "vascular": ["circulation", "blood flow", "perfusion", "ischemia", "pulse"],
                "emergency": ["life-threatening", "critical", "immediate", "emergent", "severe"],
                "orthopedic": ["fracture", "bone", "joint", "mobility", "movement"],
                "neurologist": ["neurological", "numbness", "tingling", "cognitive", "seizure"]
            }
            
            doc_notes_lower = doctor_result.notes.lower()
            for spec, keywords in specialty_keywords.items():
                if spec.lower() in doctor_result.specialty.lower():
                    found_keywords = [kw for kw in keywords if kw in doc_notes_lower]
                    if found_keywords:
                        specialty_reasoning = f" due to {', '.join(found_keywords[:2])}"
                        break
        
        clues.append(f"For complaints like '{complaint[:60]}...', consider {doctor_result.specialty}{specialty_reasoning}")
    
    # 2. Severity miscalibration with reasoning
    severity_gap = abs(ai_result.severity - doctor_result.severity)
    if severity_gap >= 2:
        errors.append("severity_under_estimated" if doctor_result.severity > ai_result.severity else "severity_over_estimated")
        
        # Extract severity reasoning from doctor notes
        severity_reasoning = ""
        if doctor_result.notes:
            # Look for severity indicators
            high_severity_indicators = ["life-threatening", "critical", "immediate", "emergent", 
                                       "systemic", "sepsis", "compromise", "failure"]
            functional_indicators = ["cannot", "unable", "loss of", "absent", "severe pain"]
            
            doc_notes_lower = doctor_result.notes.lower()
            found_indicators = []
            if any(ind in doc_notes_lower for ind in high_severity_indicators):
                found_indicators.append("life-threatening indicators")
            if any(ind in doc_notes_lower for ind in functional_indicators):
                found_indicators.append("functional loss")
            
            if found_indicators:
                severity_reasoning = f" - {', '.join(found_indicators)}"
        
        clues.append(f"Severity should be {doctor_result.severity} not {ai_result.severity}{severity_reasoning}")
    
    # 3. ENHANCED: Semantic notes analysis (not just keywords!)
    ai_notes = ai_result.notes or ""
    doctor_notes = doctor_result.notes or ""
    
    if doctor_notes and len(doctor_notes.strip()) > 10:
        # Use semantic embedding comparison
        semantic_insights = extract_semantic_insights(ai_notes, doctor_notes, embedding_model)
        
        # Learn from semantic gaps
        if semantic_insights["semantic_gap"] > 0.3:  # Significant semantic difference
            errors.append("notes_insufficient")
            
            # Extract missing concepts (semantically unique)
            if semantic_insights["missing_concepts"]:
                unique_concepts = list(set(semantic_insights["missing_concepts"]))[:5]
                clues.append(f"Doctor emphasized: {', '.join(unique_concepts)}")
            
            # Learn reasoning patterns
            if semantic_insights["reasoning_patterns"]:
                for pattern in semantic_insights["reasoning_patterns"][:2]:
                    reasoning_patterns.append(pattern)
                    clues.append(f"Clinical reasoning: {pattern[:80]}...")
            
            # Learn clinical priorities
            if semantic_insights["clinical_priorities"]:
                for priority in semantic_insights["clinical_priorities"][:2]:
                    clinical_insights.append(priority)
                    clues.append(f"Priority insight: {priority[:80]}...")
        
        # Also check length difference as secondary indicator
        notes_length_diff = len(doctor_notes) - len(ai_notes)
        if notes_length_diff > 100 and semantic_insights["semantic_gap"] < 0.3:
            # Doctor added substantial detail but semantically similar - might be elaboration
            errors.append("notes_insufficient")
            clues.append("Doctor provided more detailed clinical context - include comprehensive assessment")
    
    # Only store if there's something to learn
    if not errors:
        return
    
    # Create comprehensive lesson document with semantic understanding
    lesson_text = f"Complaint: {complaint}\nErrors: {', '.join(errors)}\nLessons: {'; '.join(clues)}"
    if reasoning_patterns:
        lesson_text += f"\nReasoning Patterns: {'; '.join(reasoning_patterns)}"
    if clinical_insights:
        lesson_text += f"\nClinical Insights: {'; '.join(clinical_insights)}"
    
    # Create multiple embeddings for better retrieval
    complaint_embedding = embedding_model.encode(complaint).tolist()
    lesson_embedding = embedding_model.encode(lesson_text).tolist()
    
    # Also embed doctor notes for semantic retrieval
    doctor_notes_embedding = embedding_model.encode(doctor_notes).tolist() if doctor_notes else None
    
    lesson_doc = {
        "case_id": case_id,
        "timestamp": datetime.now(),
        "complaint_text": complaint,
        "complaint_embedding": complaint_embedding,  # For complaint-based retrieval
        "lesson_embedding": lesson_embedding,  # For lesson-based retrieval
        "doctor_notes_embedding": doctor_notes_embedding,  # For semantic notes retrieval
        "error_types": errors,
        "lessons_learned": clues,
        "reasoning_patterns": reasoning_patterns,
        "clinical_insights": clinical_insights,
        "severity_gap": severity_gap,
        "ai_specialty": ai_result.specialty,
        "doctor_specialty": doctor_result.specialty,
        "ai_notes": ai_notes,  # Store for comparison
        "doctor_notes": doctor_notes,  # Store full doctor notes for context
        "semantic_gap": float(semantic_insights.get("semantic_gap", 0.0)) if doctor_notes else 0.0
    }
    
    lessons.insert_one(lesson_doc)
    print(f"📚 Enhanced lesson stored: {errors} | Semantic gap: {semantic_insights.get('semantic_gap', 0.0):.2f} | Patterns: {len(reasoning_patterns)}")

def get_relevant_lessons(complaint: str, limit: int = 3) -> List[Dict]:
    """
    Enhanced semantic search for relevant past mistakes using multiple embedding strategies.
    Searches by complaint similarity, lesson similarity, and doctor notes similarity.
    """
    lessons = get_collection("learning_memory")
    embedding_model = get_embedding_model()
    
    if not complaint:
        return []
    
    # Embed the current complaint
    query_embedding = np.array(embedding_model.encode(complaint))
    
    # MongoDB Atlas vector search (if available) or fallback to in-memory
    try:
        # For MongoDB Atlas with vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": query_embedding.tolist(),
                    "path": "complaint_embedding",
                    "numCandidates": 50,
                    "limit": limit * 2,  # Get more candidates for re-ranking
                    "index": "complaint_embedding_index"
                }
            }
        ]
        candidate_lessons = list(lessons.aggregate(pipeline))
    except Exception:
        # Fallback: brute-force similarity with multiple strategies
        all_lessons = list(lessons.find().sort("timestamp", -1).limit(100))
        candidate_lessons = []
        
        for lesson in all_lessons:
            # Strategy 1: Complaint similarity
            complaint_emb = np.array(lesson.get("complaint_embedding", []))
            if len(complaint_emb) > 0:
                complaint_sim = np.dot(query_embedding, complaint_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(complaint_emb)
                )
            else:
                complaint_sim = 0.0
            
            # Strategy 2: Lesson similarity (if available)
            lesson_emb = np.array(lesson.get("lesson_embedding", []))
            lesson_sim = 0.0
            if len(lesson_emb) > 0:
                lesson_sim = np.dot(query_embedding, lesson_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(lesson_emb)
                )
            
            # Strategy 3: Doctor notes similarity (if available)
            doc_notes_emb = np.array(lesson.get("doctor_notes_embedding", []))
            notes_sim = 0.0
            if len(doc_notes_emb) > 0:
                notes_sim = np.dot(query_embedding, doc_notes_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_notes_emb)
                )
            
            # Weighted combination: complaint (0.5) + lesson (0.3) + notes (0.2)
            combined_similarity = (complaint_sim * 0.5) + (lesson_sim * 0.3) + (notes_sim * 0.2)
            
            if combined_similarity > 0.55:  # Lower threshold for better recall
                lesson["_similarity"] = combined_similarity
                candidate_lessons.append(lesson)
        
        # Sort by combined similarity
        candidate_lessons.sort(key=lambda x: x.get("_similarity", 0.0), reverse=True)
        candidate_lessons = candidate_lessons[:limit * 2]
    
    # Re-rank candidates by relevance and recency
    scored_lessons = []
    for lesson in candidate_lessons:
        score = lesson.get("_similarity", 0.7)  # Default similarity
        
        # Boost recent lessons (last 7 days)
        if "timestamp" in lesson:
            days_old = (datetime.now() - lesson["timestamp"]).days
            if days_old <= 7:
                score += 0.1  # Boost recent lessons
            if days_old <= 1:
                score += 0.1  # Extra boost for very recent
        
        # Boost lessons with reasoning patterns (more valuable)
        if lesson.get("reasoning_patterns"):
            score += 0.05 * len(lesson["reasoning_patterns"])
        
        # Boost lessons with high semantic gap (more learning value)
        if lesson.get("semantic_gap", 0) > 0.4:
            score += 0.1
        
        lesson["_relevance_score"] = score
        scored_lessons.append(lesson)
    
    # Sort by relevance score and return top results
    scored_lessons.sort(key=lambda x: x.get("_relevance_score", 0.0), reverse=True)
    return scored_lessons[:limit]

_pattern_cache = None

def get_pattern_context() -> str:
    """Build global pattern context from correction statistics"""
    global _pattern_cache
    
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
    global _pattern_cache
    _pattern_cache = None

# ------------------------------------------------------------
# LangChain AI Agent
# ------------------------------------------------------------
class MedicalTriageAgent:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=Triage)
        # Lazy initialization - only create model when needed
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model to avoid startup errors"""
        if self._model is None:
            # Check both settings and environment variable
            api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Please:\n"
                    "1. Create a .env file with: OPENAI_API_KEY=your_key_here\n"
                    "2. Or set environment variable: export OPENAI_API_KEY=your_key_here"
                )
            # Ensure it's set in environment for langchain
            os.environ["OPENAI_API_KEY"] = api_key
            # ChatOpenAI will use OPENAI_API_KEY from environment automatically
            self._model = ChatOpenAI(
                model=settings.triage_model, 
                temperature=0.1, 
                max_tokens=500
            )
        return self._model
    
    async def analyze(self, desc: Optional[str], image_b64: Optional[str]) -> Triage:
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
                result = self.parser.parse(response.content)
            else:
                # Text mode
                prompt = PromptTemplate(
                    template=full_prompt,
                    input_variables=["desc"],
                    partial_variables=partial_vars
                )
                chain = prompt | self.model | self.parser
                result = chain.invoke({"desc": desc or "No description provided"})
            
            return result
                
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            print(f"Full trace: {traceback.format_exc()}")
            
            return Triage(
                specialty="General Practitioner",
                severity=2,
                notes=f"Analysis error: {str(e)} - requires manual review"
            )
    
    def _build_case_context(self, complaint: str) -> str:
        """
        Build comprehensive case-specific learning context from past mistakes.
        Includes semantic insights, reasoning patterns, and clinical priorities.
        """
        relevant_lessons = get_relevant_lessons(complaint, limit=3)
        
        if not relevant_lessons:
            return ""
        
        context = "\n\n💡 RELEVANT PAST LEARNINGS (learn from these corrections):\n"
        
        for i, lesson in enumerate(relevant_lessons, 1):
            timestamp = lesson.get('timestamp', datetime.now())
            if isinstance(timestamp, datetime):
                date_str = timestamp.strftime('%Y-%m-%d')
            else:
                date_str = str(timestamp)
            
            similarity = lesson.get("_similarity", lesson.get("_relevance_score", 0.0))
            context += f"\n{i}. Similar case ({date_str}, relevance: {similarity:.2f}):\n"
            
            # Add basic lessons
            for clue in lesson.get("lessons_learned", []):
                context += f"   - {clue}\n"
            
            # Add reasoning patterns (why doctor made the decision)
            reasoning_patterns = lesson.get("reasoning_patterns", [])
            if reasoning_patterns:
                context += f"   - Reasoning: {reasoning_patterns[0][:100]}...\n"
            
            # Add clinical insights (what doctor prioritized)
            clinical_insights = lesson.get("clinical_insights", [])
            if clinical_insights:
                context += f"   - Clinical priority: {clinical_insights[0][:100]}...\n"
            
            # Add specialty correction context
            if lesson.get("ai_specialty") != lesson.get("doctor_specialty"):
                context += f"   - Specialty correction: {lesson.get('ai_specialty')} → {lesson.get('doctor_specialty')}\n"
            
            # Add severity correction context
            if lesson.get("severity_gap", 0) >= 2:
                context += f"   - Severity gap: {lesson.get('severity_gap')} levels\n"
        
        context += "\nApply these semantic insights and reasoning patterns to the current case.\n"
        context += "Pay special attention to clinical priorities and reasoning that doctors emphasized.\n"
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

def submit_correction(case_id: str, doctor_result: Triage) -> tuple:
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
    
    return correction_score, changes

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

def cleanup_corrupted_data(clear_learning_memory: bool = False):
    """Emergency cleanup"""
    # Reset prompt
    prompts = get_collection("prompts")
    prompts.delete_many({"name": {"$in": ["current_prompt", "prompt_version"]}})
    save_prompt(_default_prompt())
    
    # Clear caches
    clear_pattern_cache()
    
    # Optionally clear learning memory
    if clear_learning_memory:
        get_collection("learning_memory").delete_many({})

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
