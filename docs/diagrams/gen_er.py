from graphviz import Digraph

g = Digraph("ERDiagram", format="png")
g.attr(rankdir="LR", fontname="Helvetica", bgcolor="white")
g.attr("node", shape="record", fontname="Helvetica", fontsize="10", style="filled", fillcolor="#FBEAEA", color="#C44E52")

g.node("STUDENT", "{STUDENTS|"
       "PK student_id: TEXT\\l"
       "branch: TEXT\\l"
       "gender: TEXT\\l"
       "attendance_pct: REAL\\l"
       "cgpa: REAL\\l"
       "backlogs: INTEGER\\l"
       "internships: INTEGER\\l"
       "projects_completed: INTEGER\\l"
       "certifications: INTEGER\\l"
       "coding_score: REAL\\l"
       "communication_score: REAL\\l"
       "extracurricular_score: REAL\\l"
       "aptitude_score: REAL\\l"
       "performance_score: REAL\\l"
       "placed: INTEGER\\l"
       "package_lpa: REAL\\l"
       "created_at: TEXT\\l"
       "updated_at: TEXT\\l}")

g.render("er_diagram", cleanup=True)
print("saved er_diagram.png")
