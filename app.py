"""
app.py
-------
Streamlit dashboard for the AI-Based Student Performance & Placement
Prediction System. Brings together the three major functional modules:

  1. Student Data Management (CRUD via src/database.py)
  2. Performance Prediction   (src/performance_model.py)
  3. Placement Prediction & Analytics (src/placement_model.py, src/visualize.py)

Run:
    streamlit run app.py
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import pandas as pd

import database as db
from performance_model import predict_performance
from placement_model import predict_placement


BRANCHES = ["CSE", "IT", "ECE", "MECH", "CIVIL", "EEE"]
GENDERS = ["Male", "Female"]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance & Placement Predictor",
    layout="wide"
)

db.init_db()

st.title(
    "🎓 AI-Based Student Performance & Placement Prediction System"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Student Data Management",
        "📈 Performance Prediction",
        "💼 Placement Prediction",
        "📊 Analytics"
    ]
)


# ---------------------------------------------------------------------------
# Module 1: Student Data Management (CRUD)
# ---------------------------------------------------------------------------

with tab1:

    st.subheader("Student Records")

    # ========================================================
    # ADD STUDENT
    # ========================================================

    with st.expander("➕ Add New Student"):

        with st.form("add_student_form"):

            c1, c2, c3 = st.columns(3)

            student_id = c1.text_input("Student ID")
            branch = c2.selectbox("Branch", BRANCHES)
            gender = c3.selectbox("Gender", GENDERS)

            c4, c5, c6 = st.columns(3)

            attendance_pct = c4.slider(
                "Attendance %",
                0.0,
                100.0,
                80.0
            )

            cgpa = c5.slider(
                "CGPA",
                0.0,
                10.0,
                7.0
            )

            backlogs = c6.number_input(
                "Backlogs",
                0,
                10,
                0
            )

            c7, c8, c9 = st.columns(3)

            internships = c7.number_input(
                "Internships",
                0,
                10,
                1
            )

            projects_completed = c8.number_input(
                "Projects Completed",
                0,
                15,
                2
            )

            certifications = c9.number_input(
                "Certifications",
                0,
                15,
                1
            )

            c10, c11, c12 = st.columns(3)

            coding_score = c10.slider(
                "Coding Score",
                0.0,
                100.0,
                65.0
            )

            communication_score = c11.slider(
                "Communication Score",
                0.0,
                100.0,
                70.0
            )

            aptitude_score = c12.slider(
                "Aptitude Score",
                0.0,
                100.0,
                68.0
            )

            extracurricular_score = st.slider(
                "Extracurricular Score",
                0.0,
                100.0,
                55.0
            )

            submitted = st.form_submit_button(
                "Add Student"
            )

            if submitted:

                try:

                    db.add_student(
                        {
                            "student_id": student_id,
                            "branch": branch,
                            "gender": gender,
                            "attendance_pct": attendance_pct,
                            "cgpa": cgpa,
                            "backlogs": backlogs,
                            "internships": internships,
                            "projects_completed": projects_completed,
                            "certifications": certifications,
                            "coding_score": coding_score,
                            "communication_score": communication_score,
                            "extracurricular_score": extracurricular_score,
                            "aptitude_score": aptitude_score,
                        }
                    )

                    st.success(
                        f"Student {student_id} added."
                    )

                except Exception as e:

                    st.error(
                        f"Could not add student: {e}"
                    )


    # ========================================================
    # VIEW STUDENTS
    # ========================================================

    students = db.list_students()

    if students:

        df_students = pd.DataFrame(
            students
        )

        st.dataframe(
            df_students,
            use_container_width=True
        )

        # ====================================================
        # DELETE STUDENT
        # ====================================================

        col_del1, col_del2 = st.columns([3, 1])

        del_id = col_del1.text_input(
            "Student ID to delete"
        )

        if col_del2.button("🗑️ Delete") and del_id:

            try:

                db.delete_student(
                    del_id
                )

                st.success(
                    f"Deleted {del_id}. Refresh to update the table."
                )

            except Exception as e:

                st.error(
                    str(e)
                )

    else:

        st.info(
            "No students yet. Add one above, or bulk-load the sample dataset via `src/database.py`."
        )


# ============================================================
# UPDATE STUDENT
# ============================================================

st.subheader("✏️ Update Student")

student_id = st.text_input(
    "Enter Student ID to Update",
    key="update_student_id"
)

if student_id:

    student = db.get_student(
        student_id
    )

    if student:

        st.success(
            f"Student found: {student_id}"
        )

        with st.form("update_student_form"):

            branch = st.text_input(
                "Branch",
                value=student["branch"]
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                index=0 if student["gender"] == "Male" else 1
            )

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(student["attendance_pct"])
            )

            cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                value=float(student["cgpa"])
            )

            backlogs = st.number_input(
                "Backlogs",
                min_value=0,
                value=int(student["backlogs"])
            )

            internships = st.number_input(
                "Internships",
                min_value=0,
                value=int(student["internships"])
            )

            projects = st.number_input(
                "Projects Completed",
                min_value=0,
                value=int(student["projects_completed"])
            )

            certifications = st.number_input(
                "Certifications",
                min_value=0,
                value=int(student["certifications"])
            )

            coding = st.number_input(
                "Coding Score",
                min_value=0.0,
                max_value=100.0,
                value=float(student["coding_score"])
            )

            communication = st.number_input(
                "Communication Score",
                min_value=0.0,
                max_value=100.0,
                value=float(student["communication_score"])
            )

            extracurricular = st.number_input(
                "Extracurricular Score",
                min_value=0.0,
                max_value=100.0,
                value=float(student["extracurricular_score"])
            )

            aptitude = st.number_input(
                "Aptitude Score",
                min_value=0.0,
                max_value=100.0,
                value=float(student["aptitude_score"])
            )

            update_button = st.form_submit_button(
                "💾 Update Student"
            )

            if update_button:

                updates = {
                    "branch": branch,
                    "gender": gender,
                    "attendance_pct": attendance,
                    "cgpa": cgpa,
                    "backlogs": backlogs,
                    "internships": internships,
                    "projects_completed": projects,
                    "certifications": certifications,
                    "coding_score": coding,
                    "communication_score": communication,
                    "extracurricular_score": extracurricular,
                    "aptitude_score": aptitude
                }

                try:

                    db.update_student(
                        student_id,
                        updates
                    )

                    st.success(
                        f"Student {student_id} updated successfully! ✅"
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Update failed: {e}"
                    )

    else:

        st.warning(
            f"No student found with ID: {student_id}"
        )


# ---------------------------------------------------------------------------
# Module 2: Performance Prediction
# ---------------------------------------------------------------------------

with tab2:

    st.subheader(
        "Predict Final Performance Score"
    )

    st.caption(
        "Enter a student's profile to predict their expected performance score (0-100)."
    )

    with st.form("perf_predict_form"):

        c1, c2, c3 = st.columns(3)

        branch = c1.selectbox(
            "Branch",
            BRANCHES,
            key="perf_branch"
        )

        gender = c2.selectbox(
            "Gender",
            GENDERS,
            key="perf_gender"
        )

        cgpa = c3.slider(
            "CGPA",
            0.0,
            10.0,
            7.0,
            key="perf_cgpa"
        )

        c4, c5, c6 = st.columns(3)

        attendance_pct = c4.slider(
            "Attendance %",
            0.0,
            100.0,
            80.0,
            key="perf_att"
        )

        backlogs = c5.number_input(
            "Backlogs",
            0,
            10,
            0,
            key="perf_back"
        )

        internships = c6.number_input(
            "Internships",
            0,
            10,
            1,
            key="perf_intern"
        )

        c7, c8, c9 = st.columns(3)

        projects_completed = c7.number_input(
            "Projects Completed",
            0,
            15,
            2,
            key="perf_proj"
        )

        certifications = c8.number_input(
            "Certifications",
            0,
            15,
            1,
            key="perf_cert"
        )

        coding_score = c9.slider(
            "Coding Score",
            0.0,
            100.0,
            65.0,
            key="perf_code"
        )

        c10, c11 = st.columns(2)

        communication_score = c10.slider(
            "Communication Score",
            0.0,
            100.0,
            70.0,
            key="perf_comm"
        )

        aptitude_score = c11.slider(
            "Aptitude Score",
            0.0,
            100.0,
            68.0,
            key="perf_apt"
        )

        extracurricular_score = st.slider(
            "Extracurricular Score",
            0.0,
            100.0,
            55.0,
            key="perf_extra"
        )

        run_perf = st.form_submit_button(
            "Predict Performance"
        )


    if run_perf:

        try:

            features = {
                "branch": branch,
                "gender": gender,
                "attendance_pct": attendance_pct,
                "cgpa": cgpa,
                "backlogs": backlogs,
                "internships": internships,
                "projects_completed": projects_completed,
                "certifications": certifications,
                "coding_score": coding_score,
                "communication_score": communication_score,
                "extracurricular_score": extracurricular_score,
                "aptitude_score": aptitude_score,

                "skill_index":
                    coding_score * 0.4
                    + communication_score * 0.3
                    + aptitude_score * 0.3,

                "experience_index":
                    internships * 2
                    + projects_completed
                    + certifications,

                "risk_flag":
                    int(
                        backlogs > 1
                        or attendance_pct < 65
                    ),
            }

            score = predict_performance(
                features
            )

            st.metric(
                "Predicted Performance Score",
                f"{score} / 100"
            )

        except FileNotFoundError:

            st.error(
                "Model not trained yet. Run `python src/performance_model.py` first."
            )


# ---------------------------------------------------------------------------
# Module 3: Placement Prediction
# ---------------------------------------------------------------------------

with tab3:

    st.subheader(
        "Predict Placement Outcome"
    )

    st.caption(
        "Enter a student's profile to predict placement likelihood and expected package (LPA)."
    )

    with st.form("place_predict_form"):

        c1, c2, c3 = st.columns(3)

        branch = c1.selectbox(
            "Branch",
            BRANCHES,
            key="pl_branch"
        )

        gender = c2.selectbox(
            "Gender",
            GENDERS,
            key="pl_gender"
        )

        cgpa = c3.slider(
            "CGPA",
            0.0,
            10.0,
            7.0,
            key="pl_cgpa"
        )

        c4, c5, c6 = st.columns(3)

        attendance_pct = c4.slider(
            "Attendance %",
            0.0,
            100.0,
            80.0,
            key="pl_att"
        )

        backlogs = c5.number_input(
            "Backlogs",
            0,
            10,
            0,
            key="pl_back"
        )

        internships = c6.number_input(
            "Internships",
            0,
            10,
            1,
            key="pl_intern"
        )

        c7, c8, c9 = st.columns(3)

        projects_completed = c7.number_input(
            "Projects Completed",
            0,
            15,
            2,
            key="pl_proj"
        )

        certifications = c8.number_input(
            "Certifications",
            0,
            15,
            1,
            key="pl_cert"
        )

        coding_score = c9.slider(
            "Coding Score",
            0.0,
            100.0,
            65.0,
            key="pl_code"
        )

        c10, c11 = st.columns(2)

        communication_score = c10.slider(
            "Communication Score",
            0.0,
            100.0,
            70.0,
            key="pl_comm"
        )

        aptitude_score = c11.slider(
            "Aptitude Score",
            0.0,
            100.0,
            68.0,
            key="pl_apt"
        )

        extracurricular_score = st.slider(
            "Extracurricular Score",
            0.0,
            100.0,
            55.0,
            key="pl_extra"
        )

        run_place = st.form_submit_button(
            "Predict Placement"
        )


    if run_place:

        try:

            features = {
                "branch": branch,
                "gender": gender,
                "attendance_pct": attendance_pct,
                "cgpa": cgpa,
                "backlogs": backlogs,
                "internships": internships,
                "projects_completed": projects_completed,
                "certifications": certifications,
                "coding_score": coding_score,
                "communication_score": communication_score,
                "extracurricular_score": extracurricular_score,
                "aptitude_score": aptitude_score,

                "skill_index":
                    coding_score * 0.4
                    + communication_score * 0.3
                    + aptitude_score * 0.3,

                "experience_index":
                    internships * 2
                    + projects_completed
                    + certifications,

                "risk_flag":
                    int(
                        backlogs > 1
                        or attendance_pct < 65
                    ),
            }

            result = predict_placement(
                features
            )

            colA, colB, colC = st.columns(3)

            colA.metric(
                "Prediction",
                result["prediction"]
            )

            colB.metric(
                "Placement Probability",
                f"{result['placement_probability'] * 100:.1f}%"
            )

            # ====================================================
            # POINT 14 — EXPECTED PACKAGE DISPLAY
            # ====================================================

            if result["prediction"] == "Placed":

                package_text = (
                    f"₹{result['expected_package_lpa']:.2f} LPA"
                )

            else:

                package_text = "N/A"

            colC.metric(
                "Expected Package",
                package_text
            )

        except FileNotFoundError:

            st.error(
                "Models not trained yet. Run `python src/placement_model.py` first."
            )


# ---------------------------------------------------------------------------
# Analytics / Reporting
# ---------------------------------------------------------------------------

with tab4:

    st.subheader(
        "Dataset Analytics"
    )

    data_path = (
        Path(__file__).resolve().parent
        / "data"
        / "students_dataset.csv"
    )

    if data_path.exists():

        df = pd.read_csv(
            data_path
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Students",
            len(df)
        )

        c2.metric(
            "Placement Rate",
            f"{df['placed'].mean() * 100:.1f}%"
        )

        c3.metric(
            "Avg CGPA",
            f"{df['cgpa'].mean():.2f}"
        )

        c4.metric(
            "Avg Package (Placed)",
            f"₹{df.loc[df['placed'] == 1, 'package_lpa'].mean():.2f} LPA"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.bar_chart(
                df["branch"].value_counts()
            )

            st.caption(
                "Students per branch"
            )

        with col2:

            st.bar_chart(
                df.groupby("branch")["placed"].mean() * 100
            )

            st.caption(
                "Placement rate (%) by branch"
            )

        st.line_chart(
            df[
                [
                    "cgpa",
                    "coding_score",
                    "aptitude_score"
                ]
            ]
            .sort_values("cgpa")
            .reset_index(drop=True)
        )

    else:

        st.warning(
            "Dataset not found. Run `python src/generate_dataset.py` first."
        )


# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================

st.subheader(
    "📊 Model Feature Importance"
)


# ------------------------------------------------------------
# Performance Model
# ------------------------------------------------------------

performance_importance_path = (
    "docs/diagrams/feature_importance_performance.png"
)

if os.path.exists(
    performance_importance_path
):

    st.markdown(
        "### 🎓 Performance Prediction — Feature Importance"
    )

    st.image(
        performance_importance_path,
        use_container_width=True
    )

else:

    st.warning(
        "Performance feature importance chart not found."
    )


# ------------------------------------------------------------
# Placement Model
# ------------------------------------------------------------

placement_importance_path = (
    "docs/diagrams/feature_importance_placement.png"
)

if os.path.exists(
    placement_importance_path
):

    st.markdown(
        "### 💼 Placement Prediction — Feature Importance"
    )

    st.image(
        placement_importance_path,
        use_container_width=True
    )

else:

    st.warning(
        "Placement feature importance chart not found."
    )