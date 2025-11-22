#!/bin/bash

# Medical Triage API Test Script
# Make sure the API is running: uvicorn app.main:app --reload

API_BASE="http://127.0.0.1:8000"

echo "=========================================="
echo "Medical Triage API Test Script"
echo "=========================================="
echo ""

# 1. Health Check
echo "1. Testing Health Check..."
curl -X GET "${API_BASE}/health" \
  -H "Content-Type: application/json" | jq .
echo ""
echo ""

# 2. Root endpoint
echo "2. Testing Root Endpoint..."
curl -X GET "${API_BASE}/" \
  -H "Content-Type: application/json" | jq .
echo ""
echo ""

# 3. Get Admin Stats
echo "3. Testing Admin Stats..."
curl -X GET "${API_BASE}/admin/stats" \
  -H "Content-Type: application/json" | jq .
echo ""
echo ""

# 4. Get Specialties
echo "4. Testing Get Specialties..."
curl -X GET "${API_BASE}/admin/specialties" \
  -H "Content-Type: application/json" | jq .
echo ""
echo ""

# 5. Get Pending Reviews
echo "5. Testing Get Pending Reviews..."
curl -X GET "${API_BASE}/triage/pending-reviews" \
  -H "Content-Type: application/json" | jq .
echo ""
echo ""

# 6. Get Analytics
echo "6. Testing Get Analytics..."
curl -X GET "${API_BASE}/triage/analytics" \
  -H "Content-Type: application/json" | jq .
echo ""
echo ""

# 7. Get Recent Cases
echo "7. Testing Get Recent Cases..."
curl -X GET "${API_BASE}/triage/recent-cases?limit=5" \
  -H "Content-Type: application/json" | jq .
echo ""
echo ""

# 8. Test Triage Assessment (Text only)
echo "8. Testing Triage Assessment (Text)..."
curl -X POST "${API_BASE}/triage/assess" \
  -H "Content-Type: multipart/form-data" \
  -F "description=Severe chest pain with shortness of breath, radiating to left arm. Started 30 minutes ago. Patient is sweating and feels nauseous." \
  -F "patient_id=TEST_PATIENT_001" | jq .
echo ""
echo ""

# 9. Test Triage Assessment (Another case)
echo "9. Testing Triage Assessment (Another case)..."
curl -X POST "${API_BASE}/triage/assess" \
  -H "Content-Type: multipart/form-data" \
  -F "description=Patient has a deep wound on the leg that won't heal. The wound is red, swollen, and has been present for 2 weeks. Patient is diabetic." \
  -F "patient_id=TEST_PATIENT_002" | jq .
echo ""
echo ""

# 10. Get Pending Reviews Again (should show new cases)
echo "10. Testing Get Pending Reviews (after new assessments)..."
curl -X GET "${API_BASE}/triage/pending-reviews" \
  -H "Content-Type: application/json" | jq .
echo ""
echo ""

# 11. Get Admin Prompt
echo "11. Testing Get Admin Prompt..."
curl -X GET "${API_BASE}/admin/prompt" \
  -H "Content-Type: application/json" | jq . | head -20
echo ""
echo ""

# 12. Get Pattern Context
echo "12. Testing Get Pattern Context..."
curl -X GET "${API_BASE}/admin/pattern-context" \
  -H "Content-Type: application/json" | jq .
echo ""
echo ""

echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Note: To test doctor validation, you need a case_id from pending reviews."
echo "Example validation command:"
echo ""
echo 'curl -X POST "${API_BASE}/triage/validate" \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"case_id": "YOUR_CASE_ID", "correct_specialty": "Emergency", "correct_severity": 5, "doctor_notes": "Confirmed emergency case"}'"'"' | jq .'
echo ""

