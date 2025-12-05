import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def create_pair_plot(df, features, group_name):
    print(f"Generating pair plot {group_name} (this may take 30-60 seconds)...")
    
    sns.pairplot(
        df[features + ['Hogwarts House']], 
        hue='Hogwarts House',
        palette={'Gryffindor': 'red', 'Slytherin': 'green', 
                'Ravenclaw': 'blue', 'Hufflepuff': 'yellow'},
        diag_kind='hist',
        plot_kws={'alpha': 0.5, 's': 10},
        diag_kws={'bins': 10, 'alpha': 0.6, 'edgecolor': 'white'}
    )
    
    print(f"Going to create the pair_plot_{group_name}.png file...")
    plt.savefig(f'pair_plot_{group_name}.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Done! Saved pair_plot_{group_name}.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py dataset_train.csv")
        sys.exit(1)

    df = pd.read_csv('dataset_train.csv')

    course_columns = [
        'Arithmancy', 'Astronomy', 'Herbology', 
        'Defense Against the Dark Arts', 'Divination',
        'Muggle Studies', 'Ancient Runes', 'History of Magic',
        'Transfiguration', 'Potions', 'Care of Magical Creatures',
        'Charms', 'Flying'
    ]

    create_pair_plot(df, course_columns, "all")
    # Features to KEEP for logistic regression
    features_to_keep = [
        'Defense Against the Dark Arts',
        'Herbology',
        'Charms',
        'Flying',
        'Ancient Runes',
        'Transfiguration',
        'Muggle Studies',
        'Divination',
        'History of Magic',
        'Potions'
    ]
    create_pair_plot(df, features_to_keep, "keep")

    # Features to DROP (for comparison/analysis)
    features_to_drop = [
        'Arithmancy',        # Homogeneous
        'Care of Magical Creatures',  # Most homogeneous
        'Astronomy'          # Redundant with DADA (corr = -1.0)
    ]

    create_pair_plot(df, features_to_drop, "drop")