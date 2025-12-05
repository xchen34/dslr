import sys
import pandas as pd
import matplotlib.pyplot as plt
import math
from itertools import combinations

def calculate_correlation(df, course1, course2):
    data = df[[course1, course2]].dropna()
    
    if len(data) < 2:
        return 0
    
    x = data[course1].values
    y = data[course2].values
    n = len(x)
    
    # Calculate means
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    
    # Calculate correlation components
    numerator = 0
    x_squared_sum = 0
    y_squared_sum = 0
    
    for i in range(n):
        x_diff = x[i] - x_mean
        y_diff = y[i] - y_mean
        numerator += x_diff * y_diff
        x_squared_sum += x_diff ** 2
        y_squared_sum += y_diff ** 2
    
    denominator = (x_squared_sum * y_squared_sum) ** 0.5
    
    return numerator / denominator if denominator != 0 else 0

def plot_selected_scatters(df, pairs, description):
    n_pairs = len(pairs)
    n_cols = 2
    n_rows = math.ceil(n_pairs / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 5))
    axes = axes.flatten()
    
    colors = {
        "Gryffindor": "red",
        "Ravenclaw": "blue",
        "Slytherin": "green",
        "Hufflepuff": "yellow"
    }
    
    for idx, (course1, course2) in enumerate(pairs):
        ax = axes[idx]
        
        for house in df['Hogwarts House'].unique():
            house_data = df[df['Hogwarts House'] == house]
            ax.scatter(
                house_data[course1],
                house_data[course2],
                alpha=0.5,
                label=house,
                color=colors[house],
                s=20  # point size
            )
        
        ax.set_xlabel(course1, fontsize=9)
        ax.set_ylabel(course2, fontsize=9)
        ax.set_title(f'{course1} vs {course2}', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    
    fig.tight_layout()
    fig.savefig(f'scatter_{description}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved graphs to scatter_{description}.png - open the file to view")





if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py dataset_train.csv")
        sys.exit(1)
    
    course_columns = [
        'Arithmancy', 'Astronomy', 'Herbology', 
        'Defense Against the Dark Arts', 'Divination',
        'Muggle Studies', 'Ancient Runes', 'History of Magic',
        'Transfiguration', 'Potions', 'Care of Magical Creatures',
        'Charms', 'Flying'
    ]

    all_pairs = list(combinations(course_columns, 2))

    print(len(all_pairs)) # Should be 78 pairs: C(n,2) = n*(n-1)/2
    
    df = pd.read_csv(sys.argv[1])

    # Calculate all correlations
    correlations = []
    for course1, course2 in all_pairs:
        corr = calculate_correlation(df, course1, course2)
        correlations.append((course1, course2, corr))
    #
    #
    
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\n" + "="*80)
    print("TOP 15 MOST CORRELATED PAIRS:")
    print("="*80)
    for i, (course1, course2, corr) in enumerate(correlations[:15], 1):
        c1_short = course1.split()[-1] if len(course1) > 20 else course1
        c2_short = course2.split()[-1] if len(course2) > 20 else course2
        
        # Interpretation
        if abs(corr) > 0.7:
            strength = "STRONG"
        elif abs(corr) > 0.4:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        direction = "positive" if corr > 0 else "negative"
        
        print(f"{i:2}. {c1_short:20} vs {c2_short:20}: {corr:+.3f} ({strength} {direction})")
    
    print("\n" + "="*80)
    print("BOTTOM 15 LEAST CORRELATED PAIRS (Nearly Independent):")
    print("="*80)
    for i, (course1, course2, corr) in enumerate(correlations[-15:], 1):
        c1_short = course1.split()[-1] if len(course1) > 20 else course1
        c2_short = course2.split()[-1] if len(course2) > 20 else course2
        print(f"{i:2}. {c1_short:20} vs {c2_short:20}: {corr:+.3f}")
    
    # top 6 correlated pairs
    top_pairs = [(c1, c2) for c1, c2, _ in correlations[:6]]
    print(f"\nPlotting top 6 most correlated pairs...")
    plot_selected_scatters(df, top_pairs, "top_6_corr_pairs")

    # last 6 correlated pairs
    bottom_pairs = [(c1, c2) for c1, c2, _ in correlations[-6:]]
    print(f"\nPlotting bottom 6 less correlated pairs...")
    plot_selected_scatters(df, bottom_pairs, "btm_6_corr_pairs")