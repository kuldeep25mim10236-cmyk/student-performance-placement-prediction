# Problem Statement

Placement and academic-performance outcomes for students depend on a wide
mix of academic, behavioural and skill-based factors — CGPA, attendance,
backlog history, internships, project work, certifications, coding
ability, communication skills and aptitude. Training & Placement Cells and
academic mentors typically assess these factors manually and reactively,
which means at-risk students (academically or placement-wise) are often
identified too late for meaningful intervention.

This project builds an **AI-based decision-support system** that ingests a
student's academic and activity profile and produces two predictive
outputs: (1) an expected final performance score, and (2) a placement
likelihood with an expected package, so that mentors can act early and
students can understand which factors most affect their outcomes.

---

## Scope of the Project

**In scope:**
- Management of student profile records (create, read, update, delete)
  via a local SQLite database
- Feature engineering from raw academic/activity attributes
- Training and evaluation of a regression model for academic performance
  prediction
- Training and evaluation of a classification model (placement likelihood)
  and a regression model (expected package) for placement prediction
- An interactive Streamlit dashboard exposing all of the above plus
  descriptive analytics (placement rate by branch, CGPA distribution,
  feature importance)
- Automated testing of the data pipeline and CRUD layer

**Out of scope:**
- Integration with a live college ERP/university database
- Real student personal data (a synthetic, statistically-realistic dataset
  is generated and used in its place)
- Deployment to a production web server / authentication & multi-tenant
  access control
- Real-time notification or alerting to mentors

---

## Target Users

- **Training & Placement Cell staff / academic mentors** — to identify
  students who may need extra support before exams or placement drives
- **Students** — to explore how changes in their profile (e.g. more
  projects, certifications, improved CGPA) could affect their predicted
  performance and placement outcome
- **Course instructors/evaluators** — reviewing this submission as a demo
  of applying ML, software engineering and data-management concepts
  together in one system

---

## High-Level Features

1. **Student Data Management** — validated CRUD operations on student
   academic/activity records, persisted in SQLite
2. **Performance Prediction** — predicts a 0–100 performance score from a
   student's academic and engagement profile using a Random Forest
   regressor
3. **Placement Prediction** — predicts placement likelihood (classifier)
   and, for likely-placed students, an expected package in LPA
   (regressor)
4. **Analytics Dashboard** — branch-wise placement rates, CGPA
   distribution, live KPIs, and model feature-importance visualizations
5. **Evaluation & Reporting** — consolidated model metrics (MAE, RMSE,
   R², Accuracy, Precision, Recall, F1, Confusion Matrix) saved for
   auditability
