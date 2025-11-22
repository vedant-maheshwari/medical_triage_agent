import streamlit as st
import requests
import base64
import io
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"
# For Docker: API_BASE_URL = "http://localhost:8000"

# === Streamlit UI (Complete with all features from full_app.py) ===
def main():
    st.set_page_config(
        page_title="Medical Triage AI Portal", 
        page_icon="🩺", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Portal",
        ["🤖 Patient Portal", "👨‍⚕️ Doctor Review", "📊 Learning Analytics", "🔧 Admin Tools"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "🟢 System Status: Active\n"
        "🧠 Learning: Enabled\n"
        "📸 Vision API: Ready"
    )
    
    if page == "🤖 Patient Portal":
        patient_portal()
    elif page == "👨‍⚕️ Doctor Review":
        doctor_review()
    elif page == "📊 Learning Analytics":
        learning_analytics()
    elif page == "🔧 Admin Tools":
        admin_tools()

def patient_portal():
    """Patient-facing portal for medical complaint input"""
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
        if not text_desc and not uploaded_file:
            st.error("❌ Please provide at least a description or image.")
            return
        
        with st.spinner("🤖 AI analyzing with learned patterns..."):
            result = call_triage_api(description=text_desc, image_b64=image_b64, image_file=uploaded_file)
            
            if result:
                display_assessment_results(result)
                # Save to session state for potential booking
                st.session_state['latest_assessment'] = result

def call_triage_api(description: Optional[str] = None, image_b64: Optional[str] = None, image_file = None) -> Optional[Dict]:
    """Call the FastAPI triage assessment endpoint"""
    try:
        data = {}
        if description:
            data["description"] = description
        
        files = None
        if image_file:
            # Use the file directly for multipart upload
            image_file.seek(0)
            files = {"image": (image_file.name, image_file.read(), image_file.type or "image/jpeg")}
            image_file.seek(0)
        
        response = requests.post(
            f"{API_BASE_URL}/triage/assess",
            data=data,
            files=files,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}. Make sure the FastAPI server is running.")
        st.info("Run: `uvicorn app.main:app --reload`")
        return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None

def display_assessment_results(result: Dict):
    """Display the AI assessment results"""
    st.success("✅ Assessment complete! A doctor will review this shortly.")
    
    st.subheader("🩺 AI Assessment Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Recommended Specialist", result['specialty'])
    with col2:
        st.metric("Severity Level", f"{result['severity']}/5")
    with col3:
        st.metric("Priority", result.get('priority', 'Unknown'))
    
    st.info(result['notes'])
    
    st.markdown("---")
    st.subheader("Recommended Action")
    
    recommended_action = result.get('recommended_action', '')
    if result['severity'] >= 4:
        st.error(f"🚨 {recommended_action}")
    elif result['severity'] == 3:
        st.warning(f"📅 {recommended_action}")
    elif result['severity'] == 2:
        st.info(f"📞 {recommended_action}")
    else:
        st.success(f"🏠 {recommended_action}")
    
    st.caption("💡 This assessment will be reviewed by a doctor for validation.")

def doctor_review():
    """Doctor review and correction interface"""
    st.title("👨‍⚕️ Doctor Review Portal")
    st.markdown("Review AI assessments and provide corrections to train the system")
    
    pending_cases = get_pending_reviews()
    
    if not pending_cases:
        st.info("✅ All assessments have been reviewed!")
        return
    
    st.subheader(f"📋 {len(pending_cases)} cases awaiting review")
    
    case_options = [
        f"Case {c['case_id'][-6:]} - {c['patient_id']} ({c['timestamp'][:10]})" 
        for c in pending_cases
    ]
    
    if not case_options:
        st.warning("No cases available")
        return
    
    case_index = st.selectbox(
        "Select case to review",
        range(len(pending_cases)),
        format_func=lambda i: case_options[i]
    )
    case = pending_cases[case_index]
    
    if case:
        display_doctor_review_form(case)

def get_pending_reviews() -> List[Dict]:
    """Get pending reviews from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/triage/pending-reviews", timeout=10)
        if response.status_code == 200:
            cases = response.json()
            # Convert datetime strings back to readable format
            for case in cases:
                if isinstance(case.get('timestamp'), str):
                    try:
                        dt = datetime.fromisoformat(case['timestamp'].replace('Z', '+00:00'))
                        case['timestamp'] = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        pass
            return cases
        st.warning(f"Failed to load reviews: {response.text}")
        return []
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}")
        return []
    except Exception as e:
        st.error(f"Error loading reviews: {str(e)}")
        return []

def display_doctor_review_form(case: Dict):
    """Display the doctor review form for a case"""
    st.markdown("---")
    st.subheader("📋 Case Details")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**👤 Patient Info**")
        st.code(f"ID: {case['patient_id']}\nTime: {case.get('timestamp', 'N/A')}")
    
    with col2:
        st.markdown("**📝 Patient Complaint**")
        if case.get('input_text'):
            st.info(case['input_text'])
        else:
            st.caption("No text description provided")
    
    # Display image if present
    if case.get('has_image'):
        st.markdown("**📸 Patient Image**")
        try:
            # Fetch image from API
            image_response = requests.get(
                f"{API_BASE_URL}/triage/case/{case['case_id']}/image",
                timeout=10
            )
            if image_response.status_code == 200:
                st.image(image_response.content, width=300, use_column_width=False)
            else:
                st.warning("Could not load image")
        except Exception as e:
            st.warning(f"Image display error: {e}")
    
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
        specialty_options = [
            "General Practitioner", "Emergency", "Cardiologist", "Orthopedic", 
            "Dermatologist", "Vascular Surgeon", "Plastic Surgeon", "Neurologist", 
            "Gastroenterologist", "Pulmonologist", "Rheumatologist", "Otolaryngologist (ENT)",
            "Ophthalmologist", "Urologist", "Gynecologist", "Pediatrician"
        ]
        
        try:
            ai_index = specialty_options.index(case['ai_specialty'])
        except ValueError:
            ai_index = 0
        
        corrected_spec = st.selectbox(
            "Corrected Specialist",
            specialty_options,
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
        submit_doctor_correction(
            case_id=case['case_id'],
            specialty=corrected_spec,
            severity=corrected_severity,
            notes=corrected_notes
        )

def submit_doctor_correction(case_id: str, specialty: str, severity: int, notes: str):
    """Submit doctor's correction to API"""
    try:
        payload = {
            "case_id": case_id,
            "correct_specialty": specialty,
            "correct_severity": severity,
            "doctor_notes": notes
        }
        
        with st.spinner("Training AI on your correction..."):
            response = requests.post(
                f"{API_BASE_URL}/triage/validate",
                json=payload,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Correction submitted! Learning score: {result['validation_score']:.3f}")
            if result.get('changes'):
                st.info("Changes: " + ", ".join(result['changes']))
            st.balloons()
            st.rerun()
        else:
            st.error(f"Failed: {response.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def learning_analytics():
    """Show agent learning progress and statistics"""
    st.title("📊 Learning Analytics Dashboard")
    
    analytics = get_learning_analytics()
    
    if not analytics:
        st.info("No data available yet. Review some cases to see analytics.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", analytics['total_feedback'])
    
    with col2:
        st.metric("Avg Accuracy", f"{analytics['avg_correction_score']*100:.1f}%")
    
    with col3:
        st.metric("Lessons Learned", analytics['total_lessons'])
    
    with col4:
        target_met = "✅" if analytics['avg_correction_score'] > 0.8 else "⏳"
        st.metric("Target (80%)", target_met)
    
    st.markdown("---")
    
    # Correction Patterns Heatmap
    if analytics.get('correction_patterns'):
        st.subheader("🎯 Correction Patterns")
        
        pattern_data = []
        for ai_spec, corrections in analytics['correction_patterns'].items():
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
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance Trend
    if analytics.get('improvement_trend'):
        st.subheader("📈 Performance Trend")
        
        trend_data = analytics['improvement_trend']
        dates = [item["date"] for item in trend_data]
        scores = [item["score"] for item in trend_data]
        
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
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent Cases Table
    st.subheader("📝 Recent Cases")
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    recent_cases = get_recent_cases()
    if recent_cases:
        # Format for display
        display_cases = []
        for case in recent_cases:
            try:
                dt = datetime.fromisoformat(case['timestamp'].replace('Z', '+00:00'))
                time_str = dt.strftime("%m-%d %H:%M")
            except:
                time_str = case.get('timestamp', 'N/A')
            
            display_cases.append({
                "Case ID": case['case_id'],
                "Patient": case['patient_id'],
                "AI Specialist": case['ai_specialty'],
                "Correction": case.get('doctor_specialty', 'None'),
                "Score": f"{case.get('correction_score', 0):.3f}",
                "Gap": case.get('severity_gap', 0),
                "Time": time_str
            })
        
        if display_cases:
            st.dataframe(pd.DataFrame(display_cases), use_container_width=True, hide_index=True)
    else:
        st.info("No recent cases available")

def get_learning_analytics() -> Optional[Dict]:
    """Get analytics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/triage/analytics", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}")
        return None
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")
        return None

def get_recent_cases(limit: int = 15) -> List[Dict]:
    """Get recent cases from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/triage/recent-cases?limit={limit}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error loading recent cases: {str(e)}")
        return []

def admin_tools():
    """Administrative tools"""
    st.title("🔧 Admin Tools")
    st.warning("⚠️ Use these tools carefully. They affect the entire system.")
    
    # Get admin stats
    stats = get_admin_stats()
    
    if stats:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prompt Versions", stats.get('prompt_versions', 0))
        
        with col2:
            total = stats.get('total_cases', 0)
            reviewed = stats.get('reviewed_cases', 0)
            st.metric("Reviews Completed", f"{reviewed}/{total}")
        
        with col3:
            st.metric("Total Lessons", stats.get('total_lessons', 0))
    
    st.markdown("---")
    
    # Health Check
    st.subheader("🔍 System Health")
    if st.button("🔍 Check System Health"):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                if health.get('status') == 'healthy':
                    st.success("✅ System is healthy")
                    st.json(health)
                else:
                    st.error(f"❌ System unhealthy: {health}")
            else:
                st.error("❌ Health check failed")
        except Exception as e:
            st.error(f"❌ Cannot connect to API: {str(e)}")
    
    st.markdown("---")
    
    # Emergency Actions
    st.subheader("🚨 Emergency Actions")
    
    if st.button("🧹 Reset System to Clean State", type="secondary"):
        clear_memory = st.checkbox("Also clear learning memory?", key="clear_memory_check")
        
        try:
            # Use query parameter for POST request
            response = requests.post(
                f"{API_BASE_URL}/admin/reset-system",
                params={"clear_learning_memory": clear_memory},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ {result.get('message', 'System reset complete!')}")
                if result.get('learning_memory_cleared'):
                    st.info("Learning memory has been cleared.")
                st.rerun()
            else:
                st.error(f"Failed: {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Maintenance
    st.subheader("🔍 Data Maintenance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Recalculate All Scores"):
            try:
                with st.spinner("Recalculating correction scores..."):
                    response = requests.post(f"{API_BASE_URL}/admin/recalculate-scores", timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Recalculated {result.get('updated_count', 0)} scores")
                else:
                    st.error(f"Failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear Pattern Cache"):
            try:
                response = requests.post(f"{API_BASE_URL}/admin/clear-cache", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Pattern cache cleared")
                    st.rerun()
                else:
                    st.error(f"Failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Debug Information
    st.subheader("📝 Debug Information")
    
    tab1, tab2 = st.tabs(["Current Base Prompt", "Pattern Context"])
    
    with tab1:
        try:
            response = requests.get(f"{API_BASE_URL}/admin/prompt", timeout=5)
            if response.status_code == 200:
                prompt_data = response.json()
                st.code(prompt_data.get('prompt', ''), language="text", wrap_lines=True)
            else:
                st.error("Failed to load prompt")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with tab2:
        try:
            response = requests.get(f"{API_BASE_URL}/admin/pattern-context", timeout=5)
            if response.status_code == 200:
                context_data = response.json()
                context = context_data.get('context', '')
                if context and context != "No patterns learned yet":
                    st.code(context, language="text")
                else:
                    st.info("No patterns learned yet. Review more cases to enable learning.")
            else:
                st.error("Failed to load pattern context")
        except Exception as e:
            st.error(f"Error: {str(e)}")

def get_admin_stats() -> Optional[Dict]:
    """Get admin statistics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/admin/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error loading admin stats: {str(e)}")
        return None

if __name__ == "__main__":
    main()
