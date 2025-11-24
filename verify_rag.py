import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_step(msg):
    print(f"\n🔹 {msg}")

def login_admin():
    print_step("Logging in as Admin...")
    response = requests.post(f"{BASE_URL}/token", data={
        "username": "admin", "password": "admin123"
    })
    if response.status_code != 200:
        print("❌ Login failed")
        exit(1)
    return response.json()["access_token"]

def submit_case(complaint):
    print_step(f"Submitting Case: '{complaint}'")
    # Endpoint uses Form data, not JSON
    response = requests.post(f"{BASE_URL}/triage/assess", data={"description": complaint})
    if response.status_code != 200:
        print(f"❌ Submission failed: {response.text}")
        return None
    data = response.json()
    print(f"✅ AI Prediction: {data['specialty']} (Severity {data['severity']})")
    return data

def get_recent_case_id(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/triage/recent-cases", headers=headers)
    cases = response.json()
    if not cases:
        print("❌ No cases found")
        return None
    return cases[0]["case_id"]

def train_ai(token, case_id, correct_specialty, notes):
    print_step(f"Training AI on Case {case_id}...")
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "case_id": case_id,
        "correct_specialty": correct_specialty,
        "correct_severity": 2,
        "doctor_notes": notes
    }
    response = requests.post(f"{BASE_URL}/admin/review/{case_id}", json=payload, headers=headers)
    if response.status_code == 200:
        print("✅ Training submitted successfully")
    else:
        print(f"❌ Training failed: {response.text}")

def check_rag(token, complaint):
    print_step(f"Checking RAG for: '{complaint}'")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/admin/prompt/preview", json={"description": complaint}, headers=headers)
    data = response.json()
    
    lessons = data.get("relevant_lessons", [])
    if lessons:
        print(f"✅ RAG Working! Found {len(lessons)} relevant lessons.")
        for i, l in enumerate(lessons):
            complaint_text = l.get('complaint_text', 'N/A')[:50]
            specialty = l.get('doctor_specialty', 'N/A')
            similarity = l.get('_similarity', 0)
            print(f"   {i+1}. [{similarity:.3f}] {complaint_text}... → {specialty}")
    else:
        print("❌ RAG Failed: No lessons found.")

def main():
    try:
        token = login_admin()
        
        # 1. Submit a tricky case
        complaint = "I have chest pain that hurts when I press on my ribs after lifting weights."
        submit_case(complaint)
        
        # 2. Get its ID
        case_id = get_recent_case_id(token)
        if not case_id: return

        # 3. Train AI (Correct it to Orthopedic)
        train_ai(token, case_id, "Orthopedic", "Chest pain reproducible by palpation is musculoskeletal.")
        
        # 4. Wait a moment for DB update
        time.sleep(3)
        
        # 5. Test RAG with a similar case
        check_rag(token, "My chest hurts when I touch it after gym.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
