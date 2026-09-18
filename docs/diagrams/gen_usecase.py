from graphviz import Digraph

g = Digraph("UseCase", format="png")
g.attr(rankdir="LR", fontname="Helvetica", bgcolor="white")

g.node("Student", "Student /\nAdmin User", shape="plaintext", fontsize="12")
g.node("Sys", "AI Prediction\nEngine", shape="plaintext", fontsize="12")

with g.subgraph(name="cluster_system") as c:
    c.attr(label="Student Performance & Placement Prediction System", style="rounded", color="black", fontsize="13")
    c.attr("node", shape="ellipse", style="filled", fillcolor="#EAF1FB", color="#4C72B0", fontname="Helvetica", fontsize="10")
    c.node("UC1", "Add / Update /\nDelete Student Record")
    c.node("UC2", "View Student\nRecords")
    c.node("UC3", "Predict Performance\nScore")
    c.node("UC4", "Predict Placement\n& Package")
    c.node("UC5", "View Analytics\nDashboard")
    c.node("UC6", "Train / Retrain\nModels")
    c.node("UC7", "Evaluate Model\nMetrics")

g.edge("Student", "UC1")
g.edge("Student", "UC2")
g.edge("Student", "UC3")
g.edge("Student", "UC4")
g.edge("Student", "UC5")
g.edge("Sys", "UC6")
g.edge("Sys", "UC7")
g.edge("UC3", "UC6", label="(include)", style="dashed")
g.edge("UC4", "UC6", label="(include)", style="dashed")
g.edge("UC7", "UC6", label="(include)", style="dashed")

g.render("usecase_diagram", cleanup=True)
print("saved usecase_diagram.png")
