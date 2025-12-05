import sys
import pandas as pd
import matplotlib.pyplot as plt
import math

course_columns = [
    'Arithmancy', 'Astronomy', 'Herbology', 
    'Defense Against the Dark Arts', 'Divination',
    'Muggle Studies', 'Ancient Runes', 'History of Magic',
    'Transfiguration', 'Potions', 'Care of Magical Creatures',
    'Charms', 'Flying'
]

def plot_all_histogram(df, course_columns):
    colors = {
        "Gryffindor": "red",
        "Ravenclaw": "blue",
        "Slytherin": "green",
        "Hufflepuff": "yellow"
    }

    n_courses = len(course_columns)
    n_cols = 3  # 3 columns
    n_rows = math.ceil(n_courses / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for idx, course in enumerate(course_columns):
        ax = axes[idx]
        for house, group_df in df.groupby("Hogwarts House"):
            ax.hist(
                group_df[course].dropna(),
                bins=10,
                alpha=0.3,
                label=house,
                color=colors.get(house)
            )

        ax.set_title(course, fontsize=10)
        ax.set_xlabel("Score", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    for idx in range(n_courses, len(axes)):
        axes[idx].axis('off')
    
    fig.tight_layout()
    fig.savefig('all_histograms.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved graphs to all_histograms.png - open the file to view")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py dataset_train.csv")
        sys.exit(1)
    
    df = pd.read_csv(sys.argv[1])

    plot_all_histogram(df, course_columns)