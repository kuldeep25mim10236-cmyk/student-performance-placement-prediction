from graphviz import Digraph

g = Digraph("Architecture", format="png")
g.attr(rankdir="TB", fontname="Helvetica", bgcolor="white", nodesep="0.5", ranksep="0.75")
g.attr("node", fontname="Helvetica", fontsize="11")
g.attr("edge", fontname="Helvetica", fontsize="9")

with g.subgraph(name="cluster_presentation") as c:
    c.attr(label="Presentation Layer", style="rounded,filled", color="#4C72B0", fillcolor="#EAF1FB", fontsize="13")
    c.node("UI", "Streamlit Dashboard (app.py)\nData Mgmt | Performance | Placement | Analytics",
           shape="box", style="rounded,filled", fillcolor="white")

with g.subgraph(name="cluster_app") as c:
    c.attr(label="Application / ML Layer", style="rounded,filled", color="#55A868", fillcolor="#EAF7EE", fontsize="13")
    c.node("PRE", "Preprocessing &\nFeature Engineering\n(preprocessing.py)", shape="box", style="rounded,filled", fillcolor="white")
    c.node("PERF", "Performance Prediction\nModel\n(performance_model.py)", shape="box", style="rounded,filled", fillcolor="white")
    c.node("PLACE", "Placement Prediction\nModel\n(placement_model.py)", shape="box", style="rounded,filled", fillcolor="white")
    c.node("EVAL", "Evaluation &\nAnalytics\n(evaluation.py / visualize.py)", shape="box", style="rounded,filled", fillcolor="white")
    c.edge("PRE", "PERF", style="invis")
    c.edge("PRE", "PLACE", style="invis")

with g.subgraph(name="cluster_data") as c:
    c.attr(label="Data Layer", style="rounded,filled", color="#C44E52", fillcolor="#FBEAEA", fontsize="13")
    c.node("CSV", "Student Dataset\n(students_dataset.csv)", shape="cylinder", style="filled", fillcolor="white")
    c.node("DB", "SQLite Database\n(students.db)", shape="cylinder", style="filled", fillcolor="white")
    c.node("MODELS", "Trained Model Artifacts\n(*.joblib)", shape="cylinder", style="filled", fillcolor="white")

g.edge("UI", "PRE", label="1. raw student input")
g.edge("UI", "DB", label="CRUD ops")
g.edge("CSV", "PRE", label="training data")
g.edge("PRE", "PERF", label="2. features")
g.edge("PRE", "PLACE", label="2. features")
g.edge("PERF", "MODELS", label="save / load")
g.edge("PLACE", "MODELS", label="save / load")
g.edge("PERF", "UI", label="3. prediction", constraint="false")
g.edge("PLACE", "UI", label="3. prediction", constraint="false")
g.edge("CSV", "DB", label="bulk load", style="dashed")
g.edge("EVAL", "MODELS", label="evaluate", style="dashed")
g.edge("EVAL", "UI", label="charts / metrics", style="dashed")

g.render("architecture_diagram", cleanup=True)
print("saved architecture_diagram.png")
