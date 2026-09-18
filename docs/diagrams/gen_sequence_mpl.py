try:
    import matplotlib  # type: ignore[import-not-found]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    import matplotlib.patches as patches  # type: ignore[import-not-found]
except ImportError as e:
    print(f"Error: matplotlib is not installed. Please install it using 'pip install matplotlib'. Details: {e}")
    exit(1)

actors = ["User", "Dashboard\n(app.py)", "Preprocessing", "Placement Model", "SQLite Database"]
x_positions = [0, 2.5, 5, 7.5, 10]

messages = [
    (0, 1, "1: submit student profile", False),
    (1, 2, "2: build_feature_matrix()", False),
    (2, 1, "3: engineered features", True),
    (1, 3, "4: predict_placement(features)", False),
    (3, 1, "5: probability, label, package", True),
    (1, 4, "6: update_student(prediction)", False),
    (4, 1, "7: ack", True),
    (1, 0, "8: display result", False),
]

n_steps = len(messages)
top_y = n_steps + 1
bottom_y = 0.3

fig, ax = plt.subplots(figsize=(11, 7))

# Lifeline boxes + dashed lines
for x, name in zip(x_positions, actors):
    ax.add_patch(patches.FancyBboxPatch(
        (x - 0.9, top_y), 1.8, 0.8,
        boxstyle="round,pad=0.05", linewidth=1.2,
        edgecolor="#4C72B0", facecolor="#EAF1FB"
    ))
    ax.text(x, top_y + 0.4, name, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.plot([x, x], [top_y, bottom_y], linestyle="--", color="gray", linewidth=1, zorder=0)

# Arrows for each message, stepping down the page
for i, (src, dst, label, is_return) in enumerate(messages):
    y = top_y - 1 - i
    x_src, x_dst = x_positions[src], x_positions[dst]
    style = "->" if not is_return else "->"
    linestyle = "dashed" if is_return else "solid"
    ax.annotate(
        "", xy=(x_dst, y), xytext=(x_src, y),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.3, linestyle=linestyle),
    )
    mid_x = (x_src + x_dst) / 2
    ax.text(mid_x, y + 0.15, label, ha="center", va="bottom", fontsize=8.5)

ax.set_xlim(-1.5, 11.5)
ax.set_ylim(bottom_y - 0.5, top_y + 1.2)
ax.axis("off")
ax.set_title("Sequence Diagram — Placement Prediction Flow", fontsize=13, fontweight="bold")

fig.tight_layout()
fig.savefig("sequence_diagram.png", dpi=150)
print("saved sequence_diagram.png")
