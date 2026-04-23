import streamlit as st
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
clf_model = joblib.load(os.path.join(BASE_DIR, "../artifacts/classification_model.pkl"))
reg_model = joblib.load(os.path.join(BASE_DIR, "../artifacts/regression_model.pkl"))


def main():
    st.set_page_config(page_title="Student Placement Predictor", layout="wide")

    st.title("🎓 Student Placement Predictor")
    st.write("AI-based prediction for placement status and expected salary")

    st.sidebar.title("App Info")
    st.sidebar.write("""
    This app predicts:
    - Placement status
    - Expected salary (if placed)
    """)

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            branch = st.selectbox("Branch", ["CSE", "ECE", "ME", "IT"])
            part_time_job = st.selectbox("Part Time Job", ["Yes", "No"])
            internet_access = st.selectbox("Internet Access", ["Yes", "No"])

            cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
            study_hours = st.slider("Study Hours/Day", 0, 12, 4)
            attendance = st.slider("Attendance %", 0, 100, 75)

        with col2:
            family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
            city_tier = st.selectbox("City Tier", ["Tier 3", "Tier 2", "Tier 1"])

            internships = st.number_input("Internships Completed", 0, 10, 1)
            projects = st.number_input("Projects Completed", 0, 20, 3)
            certifications = st.number_input("Certifications", 0, 20, 2)

            coding = st.slider("Coding Skill", 0, 10, 5)
            communication = st.slider("Communication Skill", 0, 10, 5)

        submit = st.form_submit_button("Predict")

    if submit:

        data = pd.DataFrame([{
            "gender": gender,
            "branch": branch,
            "part_time_job": part_time_job,
            "internet_access": internet_access,
            "cgpa": cgpa,
            "study_hours_per_day": study_hours,
            "attendance_percentage": attendance,
            "family_income_level": family_income,
            "city_tier": city_tier,
            "internships_completed": internships,
            "projects_completed": projects,
            "certifications_count": certifications,
            "coding_skill_rating": coding,
            "communication_skill_rating": communication,
            "tenth_percentage": 70,
            "twelfth_percentage": 70,
            "backlogs": 0,
            "aptitude_skill_rating": 5,
            "extracurricular_involvement": 1,
            "sleep_hours": 7
        }])

        expected_columns = [
            'cgpa', 'tenth_percentage', 'twelfth_percentage',
            'backlogs', 'study_hours_per_day', 'attendance_percentage',
            'projects_completed', 'internships_completed',
            'coding_skill_rating', 'communication_skill_rating',
            'aptitude_skill_rating', 'certifications_count',
            'extracurricular_involvement', 'sleep_hours',
            'family_income_level', 'city_tier',
            'gender', 'branch', 'part_time_job', 'internet_access'
        ]

        data = data.reindex(columns=expected_columns)
        data = data.fillna(0)

        with st.spinner("Predicting..."):
            try:
                pred = clf_model.predict(data)[0]
            except Exception as e:
                st.error(e)
                st.write(data)
                return

        if pred == 1:
            salary = reg_model.predict(data)[0]
            st.success(f"Placed! Estimated Salary: {salary:.2f} LPA")
            st.balloons()
        else:
            st.warning("Not Placed")


if __name__ == "__main__":
    main()
