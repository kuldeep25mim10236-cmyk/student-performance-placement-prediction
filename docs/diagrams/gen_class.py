from graphviz import Digraph

g = Digraph("ClassDiagram", format="png")
g.attr(rankdir="TB", fontname="Helvetica", bgcolor="white")
g.attr("node", shape="record", fontname="Helvetica", fontsize="10", style="filled", fillcolor="#EAF1FB", color="#4C72B0")

g.node("DB", "{StudentDatabase|+ init_db()\\l+ add_student(record)\\l+ get_student(id)\\l+ update_student(id, updates)\\l+ delete_student(id)\\l+ list_students()\\l}")
g.node("PRE", "{Preprocessor|+ load_data(path)\\l+ clean_data(df)\\l+ engineer_features(df)\\l+ build_feature_matrix(df)\\l}")
g.node("PERF", "{PerformanceModel|- model: RandomForestRegressor\\l+ train_performance_model(csv)\\l+ predict_performance(features)\\l}")
g.node("PLACE", "{PlacementModel|- classifier: RandomForestClassifier\\l- regressor: RandomForestRegressor\\l+ train_placement_models(csv)\\l+ predict_placement(features)\\l}")
g.node("EVAL", "{Evaluator|+ run_full_evaluation(csv)\\l}")
g.node("VIZ", "{Visualizer|+ plot_cgpa_distribution(df)\\l+ plot_placement_by_branch(df)\\l+ plot_feature_importance(model, cols)\\l}")
g.node("UI", "{DashboardApp|+ render_data_management_tab()\\l+ render_performance_tab()\\l+ render_placement_tab()\\l+ render_analytics_tab()\\l}")

g.edge("UI", "DB", label="uses")
g.edge("UI", "PERF", label="uses")
g.edge("UI", "PLACE", label="uses")
g.edge("UI", "VIZ", label="uses")
g.edge("PERF", "PRE", label="uses")
g.edge("PLACE", "PRE", label="uses")
g.edge("EVAL", "PERF", label="trains/evaluates")
g.edge("EVAL", "PLACE", label="trains/evaluates")

g.render("class_diagram", cleanup=True)
print("saved class_diagram.png")
