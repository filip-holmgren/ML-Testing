import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_confusion_matrix_visualization(cm):
    tp = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]
    tn = cm[1,1]

    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)
    grand_total = cm.sum()

    grid = [
        [f"Total\n{row_totals[0]} + {row_totals[1]} = {grand_total}", f"True Positive\n{col_totals[0]}", f"False Positive\n{col_totals[1]}"],
        [f"True Positive\n{row_totals[0]}", tp, fp],
        [f"False Positive\n{row_totals[1]}", fn, tn],
    ]

    color_grid = [
        ["#c9c9c950", "#78ffff60", "#1da5a560"],
        ["#ffff7860", "#78ff7860", "#ffa5a560"],
        ["#a5a51d60", "#ffa5a560", "#78ff7860"],
    ]

    fig, ax = plt.subplots()
    ax.set_xlim(0.75, 4.1)
    ax.set_ylim(0.75, 4.1)
    ax.invert_yaxis()
    ax.axis("off")

    for y in range(3):
        for x in range(3):
            val = grid[y][x]

            # default styling
            face = "white"

            # color rules
            face = color_grid[y][x]

            x_offset = 1
            y_offset = 1
            rect = patches.Rectangle((x + x_offset, y + y_offset), 1, 1, facecolor=face, edgecolor="#a2a9b1")
            ax.add_patch(rect)

            ax.text(x + x_offset + 0.5, y + y_offset  + 0.5, str(val), ha="center", va="center")

    top_rect = patches.Rectangle(
        (2, 0.75), 2, 0.25,  # x, y, width, height
        facecolor="#4ad2d260",
        edgecolor="#a2a9b1"
    )
    left_rect = patches.Rectangle(
        (0.75, 2), 0.25, 2,  # x, y, width, height
        facecolor="#d2d23d60",
        edgecolor="#a2a9b1"
    )

    ax.add_patch(top_rect)
    ax.add_patch(left_rect)

    ax.text(3, 0.865, "Predicted label",
            ha="center", va="center")
    ax.text(0.865, 3, "Actual label",
            ha="center", va="center", rotation=90)

    plt.savefig("data/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)