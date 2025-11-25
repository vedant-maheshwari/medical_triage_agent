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
import langchain
# Patch for older langchain versions or mismatch
try:
    langchain.llm_cache = None
except:
    pass

langchain.verbose = False
langchain.debug = False
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
# === Local Imports ===
from app.database import get_collection, get_mongo_client
from app.config import settings, SPECIALTY_OPTIONS
from app.models import User, UserInDB, Triage
from app.models import TriageResponse, CaseDetail, LearningAnalytics, CorrectionResponse
from app.config import settings
import cohere
from openai import OpenAI
# Initialize clients
client = OpenAI(api_key=settings.openai_api_key)
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=settings.openai_api_key)
# Global Cohere client for multilingual embeddings
_cohere_client = None

def get_cohere_client():
    """Get Cohere client (singleton)"""
    global _cohere_client
    if _cohere_client is None:
        try:
            import cohere
            _cohere_client = cohere.ClientV2(api_key=settings.cohere_api_key)
            print("✅ Cohere multilingual embedding client loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Cohere client: {e}")
    return _cohere_client

# Global embedding model instance
_embedding_model = None

def get_embedding_model():
    """
    Returns Cohere multilingual embedding function.
    
    Uses Cohere's embed-multilingual-v3.0 model which natively understands:
    - Pure Hindi (Devanagari script)
    - Hinglish (Hindi in Roman script)
    - English
    - Mixed code-switching
    
    No translation or language detection needed!
    """
    class CohereEmbeddingWrapper:
        """Wrapper to make Cohere embeddings compatible with existing RAG code"""
        def __init__(self):
            self.client = get_cohere_client()
        
        def embed_query(self, text: str) -> list:
            """Embed a single query text"""
            response = self.client.embed(
                texts=[text],
                model="embed-multilingual-v3.0",
                input_type="search_query",  # For RAG search queries
                embedding_types=["float"]
            )
            return response.embeddings.float[0]
        
        def embed_documents(self, texts: list) -> list:
            """Embed multiple documents"""
            response = self.client.embed(
                texts=texts,
                model="embed-multilingual-v3.0",
                input_type="search_document",  # For documents being indexed
                embedding_types=["float"]
            )
            return response.embeddings.float
    
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = CohereEmbeddingWrapper()
        print("✅ Multilingual embedding model loaded (Cohere embed-multilingual-v3.0)")
    return _embedding_model

# ------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------

# ------------------------------------------------------------
# Prompt Management (Pristine Base)
# ------------------------------------------------------------
def _default_prompt() -> str:
    """Immutable base prompt with core medical logic"""
    return """You are Dr. AIDEN, a highly experienced medical triage specialist. Your role is to analyze patient complaints with precision, empathy, and clinical rigor.

Task: Analyze the following wound/symptom description and return a JSON object with:
- specialty: The SINGLE PRIMARY specialist from {specialties} best suited to handle the case.
- severity: 1-5 (5=critical/life-threatening).
- notes: Comprehensive clinical rationale. If multiple specialists are relevant, list the PRIMARY one in the 'specialty' field and mention SECONDARY/REFERRAL specialists in these notes. Address any uncertainties by outlining potential scenarios.

SEVERITY SCORING CRITERIA (Strict Adherence):
- 1: Minor. Superficial, minimal intervention required (e.g., small cuts, bruises). Self-care is sufficient.
- 2: Mild. Slightly deeper or persistent, potential for infection. Requires outpatient evaluation/monitoring (e.g., non-healing minor wounds).
- 3: Moderate. Deeper tissue involvement, significant pain, or functional impact. Intervention necessary (e.g., deep lacerations, suspected fractures, spreading redness).
- 4: Severe. Significant tissue damage, systemic symptoms, or urgent risk. Urgent medical attention needed (e.g., severe burns, deep ulcers with necrosis, chest pain with risk factors).
- 5: Critical. Life-threatening, immediate intervention required (e.g., uncontrolled bleeding, anaphylaxis, stroke signs, septic shock).

GUIDANCE FOR AMBIGUITY:
- If the description is vague, outline the potential severity range based on possible scenarios.
- Explicitly state what missing information would clarify the assessment (e.g., "Duration of symptoms?", "Presence of fever?").
- Always err on the side of caution. If unsure between two severities, choose the higher one.

PRIORITIZATION RULES:
1. Life/Limb Threat → Emergency (Severity 5)
2. Vascular Compromise (circulation issues) → Vascular Surgeon
3. Functional/Bone Issues → Orthopedic
4. Skin Surface/Dermatological → Dermatologist
5. Post-Surgical Complications → Plastic Surgeon / Original Surgeon

Patient Complaint: {desc}
Return JSON: {format_instructions}"""

def get_current_prompt() -> str:
    """Load the current base prompt (never changes)"""
    return _default_prompt()

# ------------------------------------------------------------
# Learning System with Embeddings - FIXED FOR LOCAL MONGODB
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
    ai_embedding = np.array(embedding_model.embed_query(ai_notes))
    doc_embedding = np.array(embedding_model.embed_query(doctor_notes))
    
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
            doc_sent_emb = np.array(embedding_model.embed_query(doc_sent))
            
            # Check if this sentence is semantically similar to any AI sentence
            is_unique = True
            max_ai_similarity = 0.0
            for ai_sent in ai_sentences:
                ai_sent_emb = np.array(embedding_model.embed_query(ai_sent))
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

def ensure_learning_collection():
    """Ensure learning_memory collection exists with proper indexes"""
    lessons = get_collection("learning_memory")
    # Create collection if it doesn't exist (MongoDB creates on first insert)
    try:
        # Create indexes for performance
        lessons.create_index("timestamp")
        lessons.create_index("case_id")
        lessons.create_index("semantic_gap")
        print("✅ learning_memory collection verified/created with indexes")
    except Exception as e:
        print(f"⚠️ Warning: Could not create indexes: {e}")

def synthesize_lesson_with_llm(complaint: str, ai_result: Triage, doctor_result: Triage) -> Dict[str, Any]:
    """
    Use GPT-4 to deeply analyze doctor corrections and synthesize clinical lessons.
    Returns structured insights about what the AI misunderstood and what should be learned.
    """
    try:
        # Create a specialized prompt for lesson synthesis
        synthesis_prompt = f"""You are a medical education specialist analyzing AI triage corrections to extract deep clinical insights.

PATIENT COMPLAINT: {complaint}

AI'S ASSESSMENT:
- Specialty: {ai_result.specialty}
- Severity: {ai_result.severity}/5
- Notes: {ai_result.notes}

DOCTOR'S CORRECTION:
- Specialty: {doctor_result.specialty}
- Severity: {doctor_result.severity}/5
- Notes: {doctor_result.notes}

TASK: Analyze this correction and provide a structured lesson that helps the AI understand clinical reasoning.

Return a JSON object with:
{{
  "clinical_misunderstanding": "What specific clinical concept or pattern did the AI misunderstand?",
  "missed_clues": ["List of specific symptoms/signs that should have indicated the correct diagnosis"],
  "reasoning_pattern": "The clinical reasoning chain the doctor used (if X symptoms, consider Y condition because Z)",
  "severity_rationale": "Why this severity level? What makes it more/less urgent?",
  "key_learnings": ["2-3 specific, actionable lessons for similar future cases"],
  "contextual_guidance": "When you see [specific pattern], remember to [specific action] because [clinical reason]"
}}

Focus on CLINICAL REASONING, not just facts. Explain WHY, not just WHAT."""

        # Call GPT-4 for synthesis
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=800)
        response = llm.invoke(synthesis_prompt)
        
        # Parse the JSON response
        import json
        import re
        try:
            # Strip markdown code blocks if present
            content = response.content.strip()
            # Remove ```json and ``` markers
            if content.startswith("```"):
                # Extract content between code fences
                match = re.search(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL)
                if match:
                    content = match.group(1)
            
            synthesis = json.loads(content)
            print(f"✅ LLM synthesized lesson: {synthesis.get('clinical_misunderstanding', 'N/A')[:60]}...")
            return synthesis
        except json.JSONDecodeError as e:
            # Fallback: extract insights from the text response
            print(f"⚠️ LLM response JSON parsing failed: {e}")
            print(f"Raw response: {response.content[:200]}...")
            return {
                "clinical_misunderstanding": "See notes",
                "missed_clues": [],
                "reasoning_pattern": response.content[:200],
                "severity_rationale": "",
                "key_learnings": [response.content[:150]],
                "contextual_guidance": response.content[:200]
            }
    except Exception as e:
        print(f"⚠️ Error in LLM synthesis: {e}")
        # Return minimal structure so the system continues working
        return {
            "clinical_misunderstanding": "Analysis unavailable",
            "missed_clues": [],
            "reasoning_pattern": "",
            "severity_rationale": "",
            "key_learnings": [],
            "contextual_guidance": ""
        }

def store_lesson(case_id: str, complaint: str, ai_result: Triage, doctor_result: Triage):
    """
    Enhanced lesson extraction with semantic understanding and reasoning pattern learning.
    Uses embeddings to understand meaning, not just keywords.
    """
    try:
        # Ensure collection exists
        ensure_learning_collection()
        
        lessons = get_collection("learning_memory")
        embedding_model = get_embedding_model()
        
        print("\n" + "="*60)
        print("📚 STORING NEW LESSON")
        print(f"Case ID: {case_id}")
        print(f"Complaint: {complaint[:60]}...")
        print(f"AI → Doctor: {ai_result.specialty}({ai_result.severity}) → {doctor_result.specialty}({doctor_result.severity})")
        print("="*60)
        
        # === NEW: Use LLM to synthesize lesson ===
        llm_synthesis = synthesize_lesson_with_llm(complaint, ai_result, doctor_result)
        
        # Extract error types and insights from LLM synthesis
        errors = []
        clues = []
        reasoning_patterns = []
        clinical_insights = []
        
        # 1. Specialty error
        if ai_result.specialty != doctor_result.specialty:
            errors.append("specialty_wrong")
            # Use LLM's understanding of the misunderstanding
            clues.append(llm_synthesis.get("clinical_misunderstanding", ""))
            if llm_synthesis.get("missed_clues"):
                clues.extend(llm_synthesis["missed_clues"])
        
        # 2. Severity miscalibration
        severity_gap = abs(ai_result.severity - doctor_result.severity)
        if severity_gap >= 2:
            errors.append("severity_under_estimated" if doctor_result.severity > ai_result.severity else "severity_over_estimated")
            # Use LLM's severity rationale
            severity_rationale = llm_synthesis.get("severity_rationale", "")
            if severity_rationale:
                clues.append(f"Severity: {severity_rationale}")
        
        # 3. Add LLM reasoning patterns and key learnings
        if llm_synthesis.get("reasoning_pattern"):
            reasoning_patterns.append(llm_synthesis["reasoning_pattern"])
        
        if llm_synthesis.get("key_learnings"):
            for learning in llm_synthesis["key_learnings"]:
                clues.append(learning)
        
        if llm_synthesis.get("contextual_guidance"):
            clinical_insights.append(llm_synthesis["contextual_guidance"])
        
        # 4. FALLBACK: Also do semantic analysis for notes if LLM synthesis was incomplete
        ai_notes = ai_result.notes or ""
        doctor_notes = doctor_result.notes or ""
        semantic_insights = {"semantic_gap": 0.0}
        
        if doctor_notes and len(doctor_notes.strip()) > 10 and not llm_synthesis.get("key_learnings"):
            # Use semantic embedding comparison as fallback
            semantic_insights = extract_semantic_insights(ai_notes, doctor_notes, embedding_model)
            
            if semantic_insights["semantic_gap"] > 0.3:
                errors.append("notes_insufficient")
                if semantic_insights["missing_concepts"]:
                    unique_concepts = list(set(semantic_insights["missing_concepts"]))[:5]
                    clues.append(f"Doctor emphasized: {', '.join(unique_concepts)}")
        
        # Only store if there's something to learn
        if not errors:
            # Positive Reinforcement: Store correct cases to build confidence
            errors.append("positive_reinforcement")
            clues.append(f"Correctly identified {doctor_result.specialty} for these symptoms")
        
        # Create comprehensive lesson document with semantic understanding
        lesson_text = f"Complaint: {complaint}\n"
        lesson_text += f"Errors: {', '.join(errors)}\n"
        lesson_text += f"Lessons: {'; '.join(clues)}"
        
        if reasoning_patterns:
            lesson_text += f"\nReasoning Patterns: {'; '.join(reasoning_patterns)}"
        if clinical_insights:
            lesson_text += f"\nClinical Insights: {'; '.join(clinical_insights)}"
        
        # Generate embeddings
        try:
            complaint_embedding = embedding_model.embed_query(complaint)
            lesson_embedding = embedding_model.embed_query("; ".join(clues))
            doctor_notes_embedding = embedding_model.embed_query(doctor_notes) if doctor_notes else None
            
            print(f"✅ Generated embeddings (complaint: {len(complaint_embedding)}, lesson: {len(lesson_embedding)})")
        except Exception as e:
            print(f"⚠️ Error generating embeddings: {e}")
            # Use fallback embeddings to still store the lesson
            complaint_embedding = [0.01] * 1536
            lesson_embedding = [0.01] * 1536
            doctor_notes_embedding = [0.01] * 1536 if doctor_notes else None
        
        # Create lesson document
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
            "ai_severity": ai_result.severity,  # ADDED: Store AI severity
            "doctor_specialty": doctor_result.specialty,
            "doctor_severity": doctor_result.severity,  # ADDED: Store doctor severity
            "ai_notes": ai_notes,  # Store for comparison
            "doctor_notes": doctor_notes,  # Store full doctor notes for context
            "semantic_gap": float(semantic_insights.get("semantic_gap", 0.0))
        }
        
        # Store in database
        result = lessons.insert_one(lesson_doc)
        print(f"✅ Successfully stored lesson (ID: {result.inserted_id})")
        print(f"📊 Total lessons in system: {lessons.count_documents({})}")
        
        # Clear cache after storing new lesson
        clear_pattern_cache()
        
        return result.inserted_id
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR storing lesson: {e}")
        traceback.print_exc()
        return None

def detect_language(text: str) -> str:
    """
    Detect if text is primarily Hindi/Hinglish or English.
    Returns 'hi' or 'en'.
    
    Handles:
    - Pure Hindi (Devanagari script)
    - Hinglish (Hindi words written in English/Roman script)
    - Mixed code-switching
    """
    if not text:
        return "en"
    
    text_lower = text.lower()
    
    # Count Devanagari characters
    import re
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    devanagari_count = len(devanagari_pattern.findall(text))
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return "en"
    
    # If >30% Devanagari characters, definitely Hindi
    if (devanagari_count / total_chars) > 0.3:
        return "hi"
    
    # CRITICAL: Check for Hinglish (common Hindi words in Roman script)
    # This is essential for India where many people type Hindi using English letters
    hinglish_keywords = [
        # Body parts
        'haath', 'hath', 'pair', 'paon', 'per', 'सिर', 'sar', 'sir',
        'pet', 'पेट', 'kaan', 'kan', 'aankh', 'ankh', 'मुंह', 'munh', 'muh',
        'गला', 'gala', 'gardan',
        
        # Common medical terms
        'dard', 'दर्द', 'pain', 'sujan', 'suja', 'khujli', 'khaj',
        'bukhar', 'बुखार', 'fever', 'jukam', 'thand', 'khansi',
        
        # Verbs/descriptors
        'ho', 'hai', 'रहा', 'raha', 'rahi', 'mere', 'मेरे', 'mera', 'meri',
        'mujhe', 'मुझे', 'aap', 'आप', 'kya', 'क्या', 'kahan', 'kahaan', 'कहां',
        'kaisa', 'kaise', 'kyun', 'kyunki',
        
        # Time/duration
        'din', 'दिन', 'raat', 'रात', 'subah', 'shaam',
        'kal', 'aaj', 'parso',
        
        # Common phrases
        'nahi', 'नहीं', 'nahin', 'bahut', 'बहुत', 'thoda', 'jyada',
        'kaafi', 'bilkul'
    ]
    
    # Count Hinglish word matches
    words = text_lower.split()
    hinglish_matches = sum(1 for word in words if any(kw in word for kw in hinglish_keywords))
    
    # If >25% of words are Hinglish keywords, treat as Hindi
    if len(words) > 0 and (hinglish_matches / len(words)) > 0.25:
        print(f"   🇮🇳 Hinglish detected ({hinglish_matches}/{len(words)} words matched)")
        return "hi"
    
    return "en"

def translate_for_search(text: str, target_lang: str = "en") -> str:
    """
    Translate medical complaint to target language for better RAG search.
    Uses fast GPT-4o-mini for quick translation.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Fast and cheap
            temperature=0,
            api_key=settings.openai_api_key
        )
        
        prompt = f"Translate this medical complaint to {target_lang}. Only return the translation, nothing else:\n\n{text}"
        response = llm.invoke([HumanMessage(content=prompt)])
        translated = response.content.strip()
        
        return translated if translated else text
    except Exception as e:
        print(f"⚠️  Translation failed: {e}")
        return text  # Fallback to original

def check_semantic_relevance_llm(query: str, lesson_complaint: str, lesson_notes: str = "") -> float:
    """
    Use LLM to determine if lesson is relevant to query based on medical context.
    Returns relevance score 0.0-1.0
    
    This is MUCH more robust than keyword matching:
    - Works in any language
    - Understands medical synonyms
    - Handles complex descriptions
    - No fixed keyword list needed
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.openai_api_key
        )
        
        # Quick relevance check
        prompt = f"""You are a medical triage expert. Determine if these two medical complaints are about the SAME or SIMILAR medical issue/body region.

Query: "{query}"
Past Case: "{lesson_complaint}"

Consider:
- Same body part/region (e.g., both about hand, or both about chest)
- Similar symptoms (e.g., both involve pain, swelling, infection)
- Related medical conditions

Return ONLY a number 0.0-1.0:
- 1.0 = Highly relevant (same body part and similar symptoms)
- 0.7-0.9 = Related (same region or similar condition)
- 0.3-0.6 = Somewhat related (might be useful)
- 0.0-0.2 = Not relevant (different body part/condition)

Score:"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        score_text = response.content.strip()
        
        # Parse score
        try:
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to 0-1
        except:
            # If can't parse, return neutral
            return 0.5
            
    except Exception as e:
        print(f"⚠️  LLM relevance check failed: {e}")
        return 0.5  # Neutral score on failure

def get_relevant_lessons(complaint: str, limit: int = 3) -> List[Dict]:
    """
    Retrieve relevant past cases using vector similarity search.
    Enhanced with LLM-based semantic relevance checking and higher threshold for better relevance.
    
    Returns:
        List of lesson documents sorted by relevance (with _similarity score)
    """
    if not complaint or len(complaint.strip()) < 3:
        return []
    
    print(f"\n🔍 RAG SEARCH: Looking for lessons similar to: '{complaint[:50]}...'")
    
    try:
        # Ensure collection exists
        ensure_learning_collection()
        
        lessons = get_collection("learning_memory")
        embedding_model = get_embedding_model()
        
        # MULTILINGUAL EMBEDDINGS: No translation needed!
        # Cohere's embed-multilingual-v3.0 natively understands:
        # - Hindi (Devanagari): "मेरे हाथ में दर्द है"
        # - Hinglish: "mere haath me dard hai"
        # - English: "pain in my hand"
        # All are semantically matched without any translation!
        
        query_embedding = embedding_model.embed_query(complaint)
        query_vec = np.array(query_embedding)
        print(f"✅ Generated multilingual query embedding (dimension: {len(query_embedding)})")
        
        # Get recent lessons with embeddings (limit to 200 most recent for performance)
        all_lessons = list(lessons.find(
            {"complaint_embedding": {"$exists": True, "$ne": None}},
            {
                "_id": 1, "case_id": 1, "complaint_text": 1, "lessons_learned": 1, 
                "complaint_embedding": 1, "timestamp": 1, "ai_specialty": 1, 
                "doctor_specialty": 1, "reasoning_patterns": 1, "clinical_insights": 1,
                "ai_severity": 1, "doctor_severity": 1, "semantic_gap": 1, "doctor_notes": 1
            }
        ).sort("timestamp", -1).limit(200))
        
        if not all_lessons:
            print("📭 No lessons with embeddings found in database")
            # Check if collection exists but has no valid documents
            total_docs = lessons.count_documents({})
            if total_docs > 0:
                print(f"⚠️  Collection has {total_docs} documents but none with valid embeddings")
            return []
        
        print(f"📚 Found {len(all_lessons)} lessons with embeddings to compare against")
        
        # Calculate similarities with proper error handling
        scored_lessons = []
        query_norm = np.linalg.norm(query_vec) + 1e-8  # Prevent division by zero
        
        for lesson in all_lessons:
            try:
                # Get lesson embedding
                lesson_emb = np.array(lesson.get("complaint_embedding", []))
                if len(lesson_emb) == 0 or lesson_emb.shape[0] != query_vec.shape[0]:
                    continue  # Skip invalid embeddings
                    
                # Calculate cosine similarity
                lesson_norm = np.linalg.norm(lesson_emb) + 1e-8
                similarity = np.dot(query_vec, lesson_emb) / (query_norm * lesson_norm)
                
                # CRITICAL: Lowered threshold to 0.65 for better Hinglish/Hindi matching
                # The LLM semantic check below will filter out false positives
                if similarity < 0.65:
                    continue
                
                # PHASE 2: LLM-based semantic relevance check (robust for any language/description)
                # Uses GPT-4o-mini to determine if lesson is about same medical issue
                llm_relevance = check_semantic_relevance_llm(
                    query=complaint,  # Use original query (any language)
                    lesson_complaint=lesson.get('complaint_text', ''),
                    lesson_notes=lesson.get('doctor_notes', '')
                )
                
                # Combined score: embedding similarity + LLM relevance
                combined_score = (similarity * 0.6) + (llm_relevance * 0.4)
                
                if llm_relevance < 0.5:  # LLM says not relevant
                    print(f"⏭️  Skipping (LLM relevance: {llm_relevance:.2f}): {lesson.get('complaint_text', '')[:50]}... (embedding: {similarity:.3f})")
                    continue
                    
                # Add combined score to lesson (embedding + LLM relevance)
                lesson["_similarity"] = float(combined_score)
                lesson["_embedding_similarity"] = float(similarity)
                lesson["_llm_relevance"] = float(llm_relevance)
                scored_lessons.append(lesson)
                print(f"✅ Match found (combined: {combined_score:.3f}, embed: {similarity:.3f}, LLM: {llm_relevance:.2f}): {lesson.get('complaint_text', '')[:50]}...")
                
            except Exception as e:
                print(f"⚠️  Error processing lesson {lesson.get('case_id', 'unknown')}: {e}")
                continue
        
        if not scored_lessons:
            print("❌ No relevant lessons found above similarity threshold (0.70)")
            # Debug: Print top 3 rejected scores to help tuning
            try:
                debug_scores = []
                for lesson in all_lessons:
                    l_emb = np.array(lesson.get("complaint_embedding", []))
                    if len(l_emb) > 0:
                        sim = np.dot(query_vec, l_emb) / (query_norm * (np.linalg.norm(l_emb) + 1e-8))
                        debug_scores.append((sim, lesson.get('complaint_text', '')))
                debug_scores.sort(key=lambda x: x[0], reverse=True)
                print("   Top 3 rejected scores:")
                for s, t in debug_scores[:3]:
                    print(f"   - {s:.3f}: {t[:50]}...")
            except: pass
            return []
        
        # Sort by similarity and apply recency boost
        scored_lessons.sort(key=lambda x: x["_similarity"], reverse=True)
        
        # Apply recency boost (within last 30 days) - ONLY for relevant lessons
        now = datetime.now()
        for lesson in scored_lessons:
            days_old = (now - lesson["timestamp"]).days if "timestamp" in lesson else 999
            if days_old <= 30:
                # Reduced boost from 0.15 to 0.05 to prevent overriding semantic relevance
                lesson["_relevance_score"] = lesson["_similarity"] + (0.05 * (30 - days_old) / 30)
            else:
                lesson["_relevance_score"] = lesson["_similarity"]
        
        # Sort by final relevance score
        scored_lessons.sort(key=lambda x: x["_relevance_score"], reverse=True)
        
        print(f"🎯 Found {len(scored_lessons)} relevant lessons, returning top {limit}")
        for i, lesson in enumerate(scored_lessons[:limit], 1):
            print(f"   {i}. Relevance: {lesson['_relevance_score']:.3f} | Case: {lesson.get('complaint_text', '')[:50]}...")
            print(f"      Specialty: {lesson.get('ai_specialty', 'N/A')} → {lesson.get('doctor_specialty', 'N/A')}")
        
        return scored_lessons[:limit]
        
    except Exception as e:
        print(f"❌ Critical error in RAG search: {e}")
        traceback.print_exc()
        return []

_pattern_cache = None
def get_pattern_context() -> str:
    """Build global pattern context from correction statistics"""
    global _pattern_cache
    patterns = get_correction_patterns()
    if not patterns:
        return ""
    context = "\n🌐 GLOBAL CORRECTION PATTERNS (last 7 days):\n"
    total_corrections = sum(sum(c.values()) for c in patterns.values() if c != "NOTES_INSUFFICIENCY")
    context += f"Total corrections: {total_corrections}\n"
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
                
                # Add specific vision context to reduce refusals
                vision_instruction = (
                    "\n\nIMAGE ANALYSIS INSTRUCTION:\n"
                    "Analyze the attached medical image for clinical triage purposes only. "
                    "Focus on objective visual findings (redness, swelling, wound depth, tissue state). "
                    "This is for a professional medical triage system. Do not refuse based on gore/graphic content as this is a medical context."
                )
                
                messages = [
                    SystemMessage(content="You are Dr. AIDEN. Always output JSON."),
                    HumanMessage(content=[
                            {
                                "type": "text",
                                "text": f"{formatted_prompt}{vision_instruction}\nReturn ONLY valid JSON."
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
                try:
                    result = self.parser.parse(response.content)
                except Exception:
                    # Fallback if model refuses or returns bad JSON
                    print(f"⚠️ LLM Parsing Error/Refusal: {response.content}")
                    # Try to extract text if it's a refusal
                    refusal_text = response.content[:200].replace("\n", " ")
                    return Triage(
                        specialty="General Practitioner",
                        severity=2,
                        notes=f"AI could not process the request. Model response: '{refusal_text}'. Please review manually."
                    )
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
        
        context = "\n💡 RELEVANT PAST LEARNINGS (learn from these corrections):\n"
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
        "learned_from": False,
        "clinical_status": "pending",
        "ai_status": "pending_review",
        "assigned_to": None,
        "doctor_notes": None,
        "admin_feedback": None
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
            "input_image": 1, "clinical_status": 1, "assigned_to": 1
        }
    ).sort("timestamp", -1)
    return list(cursor)

def assign_case(case_id: str, doctor_username: str):
    """Assign a case to a doctor (lock it)"""
    feedback = get_collection("feedback")
    # Check if already assigned
    case = feedback.find_one({"_id": ObjectId(case_id)})
    if not case:
        raise ValueError("Case not found")
    if case.get("assigned_to") and case.get("assigned_to") != doctor_username:
        raise ValueError(f"Case already assigned to {case.get('assigned_to')}")
    feedback.update_one(
        {"_id": ObjectId(case_id)},
        {"$set": {
            "assigned_to": doctor_username,
            "clinical_status": "in_progress"
        }}
    )
    return True

def resolve_case(case_id: str, notes: str):
    """Mark case as treated (Clinical Workflow)"""
    feedback = get_collection("feedback")
    feedback.update_one(
        {"_id": ObjectId(case_id)},
        {"$set": {
            "clinical_status": "treated",
            "doctor_notes": notes,
            "assigned_to": None # Release assignment so it doesn't clutter
        }}
    )
    return True

def get_user_by_username(username: str) -> Optional[UserInDB]:
    """Get user by username"""
    users = get_collection("users")
    user_doc = users.find_one({"username": username})
    if user_doc:
        return UserInDB(**user_doc)
    return None

def create_user(user: UserInDB):
    """Create a new user"""
    users = get_collection("users")
    if get_user_by_username(user.username):
        raise ValueError("Username already registered")
    user_dict = user.dict()
    users.insert_one(user_dict)
    return user

def submit_correction(case_id: str, doctor_result: Triage, is_admin: bool = False) -> tuple:
    """Submit correction and trigger learning (AI Workflow)"""
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
    update_data = {
        "doctor_specialty": doctor_result.specialty,
        "doctor_severity": doctor_result.severity,
        "correction_score": correction_score,
        "feedback_given": True,
        "learned_from": True,
        "severity_gap": severity_gap,
        "ai_status": "reviewed"
    }
    
    if is_admin:
        update_data["admin_feedback"] = doctor_result.notes
    else:
        # If doctor gives feedback, we store it but maybe don't mark AI status as fully reviewed if we want admin to double check?
        # For now, let's say doctor feedback also counts as review, but admin can override.
        # Actually, per plan: "Admins will take over the responsibility of detailed AI feedback/training."
        # So if doctor gives feedback, it's optional.
        update_data["doctor_notes"] = doctor_result.notes
    
    feedback.update_one(
        {"_id": ObjectId(case_id)},
        {"$set": update_data}
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

def resolve_and_train(case_id: str, specialty: str, severity: int, notes: str, is_admin: bool = False):
    """
    Combined action: Mark case as treated AND train the AI.
    This allows doctors to contribute training data when treating patients.
    """
    # First, mark as treated
    feedback = get_collection("feedback")
    feedback.update_one(
        {"_id": ObjectId(case_id)},
        {"$set": {
            "clinical_status": "treated",
            "assigned_to": None
        }}
    )
    
    # Then, submit correction to train AI (reuse existing logic)
    correction_score, changes = submit_correction(
        case_id=case_id,
        specialty=specialty,
        severity=severity,
        notes=notes,
        is_admin=is_admin
    )
    
    print(f"✅ Case {case_id} marked as treated AND used for training")
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
    
    # ENSURE LEARNING MEMORY COLLECTION EXISTS
    ensure_learning_collection()
    
    print("✅ System initialized and ready")
    
    # Run RAG verification
    # verify_rag_system()

def analyze_triage(text: str, image_data: Optional[str] = None, language: str = "en") -> Triage:
    """
    Main triage analysis function.
    Combines:
    1. Base medical knowledge (Prompt)
    2. Learned lessons (RAG)
    3. Multi-modal input (Text + Image)
    4. Language support (English/Hindi)
    """
    # 1. Retrieve relevant lessons
    relevant_lessons = get_relevant_lessons(text)
    
    # 2. Construct dynamic prompt with language instruction
    system_prompt = _construct_system_prompt(relevant_lessons)
    
    # Add language-specific instruction
    language_instruction = ""
    if language == "hi":
        language_instruction = "\n\nIMPORTANT: The patient has communicated in Hindi. You MUST respond in Hindi (Devanagari script) while maintaining all medical accuracy. Translate specialty names and clinical notes to Hindi."
    elif language == "en":
        language_instruction = "\n\nRespond in English as specified."
    
    system_prompt += language_instruction
    
    # 3. Call LLM
    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0.0,
        api_key=settings.openai_api_key
    )
    parser = PydanticOutputParser(pydantic_object=Triage)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Patient Complaint: {text}")
    ]
    if image_data:
        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": "Here is an image of the patient's condition:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ]
            )
        )
    
    try:
        response = llm.invoke(messages)
        # If response is already a Triage object (some langchain versions do this automatically with structured output)
        if isinstance(response, Triage):
            return response
        # Otherwise parse content
        content = response.content
        # Clean markdown if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return parser.parse(content)
    except Exception as e:
        print(f"Error in analyze_triage: {e}")
        traceback.print_exc()
        # Fallback
        return Triage(
            specialty="General Practitioner", 
            severity=3, 
            notes=f"AI Analysis Failed: {str(e)}. Please review manually."
        )

def _construct_system_prompt(lessons: List[Dict]) -> str:
    """Construct dynamic system prompt with few-shot examples"""
    # Get base prompt from DB or default
    prompts = get_collection("prompts")
    current = prompts.find_one({"name": "current_prompt"})
    base_prompt = current["prompt"] if current else _default_prompt()
    
    if not lessons:
        return base_prompt
    
    # Add few-shot examples
    examples_text = "\nRELEVANT PAST CASES (Use as reference):\n"
    for i, lesson in enumerate(lessons, 1):
        examples_text += f"\nCase {i}:\n"
        examples_text += f"Complaint: {lesson.get('complaint_text', 'N/A')}\n"
        examples_text += f"Correct Triage: {lesson.get('doctor_specialty', 'N/A')} (Severity {lesson.get('doctor_severity', 'N/A')})\n"
        # Use lessons_learned or doctor_notes for reasoning
        reasoning = lesson.get('doctor_notes', '') or '; '.join(lesson.get('lessons_learned', []))
        examples_text += f"Reasoning: {reasoning}\n"
    
    return base_prompt + examples_text

def verify_rag_system():
    """Verify RAG system is working correctly"""
    print("\n" + "="*70)
    print("🔍 RAG SYSTEM VERIFICATION")
    print("="*70)
    
    # Test cases
    test_cases = [
        "Patient has a wound on their leg that won't heal after 3 weeks",
        "Deep wound on foot with redness spreading around it",
        "Chest pain when lifting weights that hurts when pressing on ribs"
    ]
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: '{test_case}'")
        lessons = get_relevant_lessons(test_case, limit=2)
        if lessons:
            print(f"✅ Found {len(lessons)} relevant lessons")
            for i, lesson in enumerate(lessons, 1):
                similarity = lesson.get('_similarity', 0)
                print(f"  {i}. Similarity: {similarity:.3f}")
                print(f"     Complaint: {lesson.get('complaint_text', '')[:60]}...")
                print(f"     Specialty Correction: {lesson.get('ai_specialty', 'N/A')} → {lesson.get('doctor_specialty', 'N/A')}")
                print(f"     Lesson: {'; '.join(lesson.get('lessons_learned', []))[:80]}...")
        else:
            print("❌ No lessons found - system may need bootstrap data")
    
    print("\n" + "="*70)
    print("✅ RAG SYSTEM VERIFICATION COMPLETE")
    print("="*70)

def delete_case(case_id: str) -> bool:
    """Delete a case from the feedback collection"""
    try:
        feedback = get_collection("feedback")
        result = feedback.delete_one({"_id": ObjectId(case_id)})
        return result.deleted_count > 0
    except Exception as e:
        print(f"❌ Error deleting case {case_id}: {e}")
        return False