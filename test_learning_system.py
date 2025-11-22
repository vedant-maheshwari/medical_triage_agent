#!/usr/bin/env python3
"""
Test script to verify the enhanced learning system with semantic understanding.
This script demonstrates how the system learns from doctor feedback.
"""

import requests
import json
import time
from datetime import datetime

API_BASE = "http://127.0.0.1:8000"

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_case(description, image=None, patient_id=None):
    """Submit a test case and return the assessment"""
    print(f"\n📝 Submitting case: {description[:60]}...")
    
    data = {}
    if description:
        data["description"] = description
    if patient_id:
        data["patient_id"] = patient_id
    
    files = None
    if image:
        files = {"image": open(image, "rb")}
    
    response = requests.post(
        f"{API_BASE}/triage/assess",
        data=data,
        files=files,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Assessment ID: {result['assessment_id']}")
        print(f"   Specialty: {result['specialty']}")
        print(f"   Severity: {result['severity']}/5")
        print(f"   Notes: {result['notes'][:100]}...")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

def submit_correction(case_id, specialty, severity, notes):
    """Submit doctor correction"""
    print(f"\n👨‍⚕️ Submitting correction...")
    print(f"   Specialty: {specialty}")
    print(f"   Severity: {severity}/5")
    print(f"   Notes: {notes[:100]}...")
    
    payload = {
        "case_id": case_id,
        "correct_specialty": specialty,
        "correct_severity": severity,
        "doctor_notes": notes
    }
    
    response = requests.post(
        f"{API_BASE}/triage/validate",
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Correction submitted!")
        print(f"   Validation Score: {result['validation_score']:.3f}")
        if result.get('changes'):
            print(f"   Changes: {', '.join(result['changes'])}")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

def get_pending_reviews():
    """Get pending reviews"""
    response = requests.get(f"{API_BASE}/triage/pending-reviews", timeout=10)
    if response.status_code == 200:
        return response.json()
    return []

def get_analytics():
    """Get learning analytics"""
    response = requests.get(f"{API_BASE}/triage/analytics", timeout=10)
    if response.status_code == 200:
        return response.json()
    return None

def main():
    print_section("Enhanced Learning System Test Suite")
    print("\nThis test demonstrates:")
    print("1. Semantic understanding (not just keywords)")
    print("2. Reasoning pattern extraction")
    print("3. Clinical priority learning")
    print("4. Multi-strategy retrieval (RAG)")
    
    # Test 1: Initial case - AI makes a mistake
    print_section("TEST 1: Initial Assessment (AI makes mistake)")
    
    case1 = test_case(
        description="Patient has deep wound on diabetic foot. Wound is red, swollen, patient has fever and feels unwell. Wound has been present for 2 weeks.",
        patient_id="TEST_LEARN_001"
    )
    
    if not case1:
        print("❌ Failed to get assessment. Is the API running?")
        return
    
    print(f"\n📊 AI Assessment:")
    print(f"   Specialty: {case1['specialty']}")
    print(f"   Severity: {case1['severity']}/5")
    print(f"   Notes: {case1['notes']}")
    
    # Test 2: Doctor corrects with semantic reasoning
    print_section("TEST 2: Doctor Correction (with semantic reasoning)")
    
    doctor_notes = """Patient has diabetic foot ulcer with signs of systemic infection. 
    Presence of fever, erythema, and systemic symptoms indicates sepsis risk. 
    Requires immediate emergency evaluation due to potential limb-threatening 
    ischemia and risk of systemic complications. Vascular assessment needed 
    but patient is too unstable for routine vascular clinic."""
    
    correction1 = submit_correction(
        case_id=case1['assessment_id'],
        specialty="Emergency",
        severity=5,
        notes=doctor_notes
    )
    
    if correction1:
        print("\n💡 Learning Points:")
        print("   - System should extract: 'systemic infection', 'sepsis risk'")
        print("   - Reasoning pattern: 'indicates sepsis risk' → 'requires immediate emergency'")
        print("   - Clinical priority: 'limb-threatening', 'systemic complications'")
        print("   - Semantic gap should be high (AI missed systemic involvement)")
    
    time.sleep(2)  # Wait for learning to process
    
    # Test 3: Similar case - AI should learn
    print_section("TEST 3: Similar Case (AI should apply learning)")
    
    case2 = test_case(
        description="Diabetic patient with foot wound. Wound is infected, patient has high fever, feels very sick. Wound won't heal.",
        patient_id="TEST_LEARN_002"
    )
    
    if case2:
        print(f"\n📊 AI Assessment (After Learning):")
        print(f"   Specialty: {case2['specialty']}")
        print(f"   Severity: {case2['severity']}/5")
        print(f"   Notes: {case2['notes']}")
        
        print(f"\n🔍 Verification:")
        if "Emergency" in case2['specialty']:
            print("   ✅ AI learned to route to Emergency for systemic infection")
        if case2['severity'] >= 4:
            print("   ✅ AI learned to increase severity for systemic involvement")
        if any(word in case2['notes'].lower() for word in ['systemic', 'sepsis', 'infection', 'emergency']):
            print("   ✅ AI learned to include systemic/sepsis concerns")
    
    # Test 4: Another learning case - compartment syndrome
    print_section("TEST 4: Compartment Syndrome Learning")
    
    case3 = test_case(
        description="Patient has severe leg pain after injury. Cannot move leg, leg feels tight and swollen.",
        patient_id="TEST_LEARN_003"
    )
    
    if case3:
        print(f"\n📊 AI Assessment:")
        print(f"   Specialty: {case3['specialty']}")
        print(f"   Severity: {case3['severity']}/5")
        
        # Doctor correction with reasoning
        doctor_notes2 = """Patient shows signs of compartment syndrome with absent 
        distal pulses and loss of sensation. Requires immediate emergency fasciotomy 
        due to risk of permanent limb loss and muscle necrosis. This is a surgical 
        emergency that cannot wait for routine orthopedic evaluation."""
        
        correction2 = submit_correction(
            case_id=case3['assessment_id'],
            specialty="Emergency",
            severity=5,
            notes=doctor_notes2
        )
        
        if correction2:
            print("\n💡 Learning Points:")
            print("   - System should extract: 'compartment syndrome', 'absent pulses'")
            print("   - Reasoning: 'absent pulses' → 'requires immediate fasciotomy'")
            print("   - Priority: 'surgical emergency', 'risk of permanent limb loss'")
    
    time.sleep(2)
    
    # Test 5: Similar compartment syndrome case
    print_section("TEST 5: Similar Compartment Syndrome Case")
    
    case4 = test_case(
        description="Patient injured leg, now cannot move it. Leg is very swollen and tight. No feeling in foot.",
        patient_id="TEST_LEARN_004"
    )
    
    if case4:
        print(f"\n📊 AI Assessment (After Learning):")
        print(f"   Specialty: {case4['specialty']}")
        print(f"   Severity: {case4['severity']}/5")
        print(f"   Notes: {case4['notes']}")
        
        print(f"\n🔍 Verification:")
        if "Emergency" in case4['specialty']:
            print("   ✅ AI learned compartment syndrome → Emergency")
        if case4['severity'] == 5:
            print("   ✅ AI learned to set severity 5 for surgical emergencies")
        if any(word in case4['notes'].lower() for word in ['compartment', 'pulse', 'fasciotomy', 'emergency']):
            print("   ✅ AI learned to consider compartment syndrome and pulses")
    
    # Test 6: Check analytics
    print_section("TEST 6: Learning Analytics")
    
    analytics = get_analytics()
    if analytics:
        print(f"\n📊 Learning Statistics:")
        print(f"   Total Feedback: {analytics.get('total_feedback', 0)}")
        print(f"   Avg Correction Score: {analytics.get('avg_correction_score', 0):.3f}")
        print(f"   Total Lessons Learned: {analytics.get('total_lessons', 0)}")
        
        if analytics.get('correction_patterns'):
            print(f"\n   Correction Patterns:")
            for ai_spec, corrections in list(analytics['correction_patterns'].items())[:3]:
                for doc_spec, count in corrections.items():
                    print(f"      {ai_spec} → {doc_spec}: {count}x")
    
    # Test 7: Check pending reviews
    print_section("TEST 7: Pending Reviews")
    
    pending = get_pending_reviews()
    print(f"\n📋 Pending Reviews: {len(pending)}")
    for i, case in enumerate(pending[:3], 1):
        print(f"\n   Case {i}:")
        print(f"      ID: {case['case_id'][-6:]}")
        print(f"      Patient: {case['patient_id']}")
        print(f"      AI Specialty: {case['ai_specialty']}")
        print(f"      AI Severity: {case['ai_severity']}/5")
    
    print_section("Test Complete!")
    print("\n✅ To verify semantic learning:")
    print("   1. Check MongoDB 'learning_memory' collection for:")
    print("      - 'reasoning_patterns' field (should contain doctor reasoning)")
    print("      - 'clinical_insights' field (should contain priorities)")
    print("      - 'semantic_gap' field (should be > 0.3 for significant differences)")
    print("      - 'doctor_notes_embedding' field (should exist)")
    print("\n   2. Check that similar cases retrieve these lessons")
    print("   3. Check that AI applies learned reasoning in new cases")
    print("\n📝 Run: python test_learning_system.py")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API. Make sure it's running:")
        print("   uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

