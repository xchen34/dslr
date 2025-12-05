import sys
import pandas as pd
import math

# ---------
# Calculation functions for describe's statistic value
# ---------
def calculate_count(series):
    return len(series.dropna())

def calculate_mean(series):
    clean_data = series.dropna()
    return sum(clean_data) / len(clean_data) if len(clean_data) > 0 else None

def calculate_std(series):
    mean = calculate_mean(series)
    clean_data = series.dropna()
    sum = 0
    count = 0
    for value in clean_data:
        sum += (value - mean) ** 2
        count += 1
    return math.sqrt(sum/(count - 1)) if count > 1 else None

def calculate_min(series):
    clean_data = series.dropna()
    if len(clean_data) == 0:
        return None
    min_val = clean_data.iloc[0]
    for value in clean_data:
        if value < min_val:
            min_val = value
    return min_val

def calculate_max(series):
    clean_data = series.dropna()
    if len(clean_data) == 0:
        return None
    max_val = clean_data.iloc[0]
    for value in clean_data:
        if value > max_val:
            max_val = value
    return max_val

def calculate_percentile(series, percentile):
    """
    Calculate percentile (0-1, e.g., 0.25 for 25th percentile)
    """
    clean_data = series.dropna().sort_values().reset_index(drop=True)
    count = len(clean_data)
    if count == 0:
        return None
    
    position = percentile * (count - 1)
    lower_index = int(position)
    upper_index = lower_index + 1
    
    if upper_index >= count:
        return clean_data.iloc[lower_index]
    
    weight = position - lower_index
    return clean_data.iloc[lower_index] * (1 - weight) + clean_data.iloc[upper_index] * weight

def chunk_by_size(lst, size):
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def print_describe_table(stats):
    # Column names
    columns = list(stats.keys())
    stat_names = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    stat_keys = ["count", "mean", "std", "min", "p25", "p50", "p75", "max"]
    
    for group in chunk_by_size(columns, 4):
        # Print header
        print(f"{'':>10} " + " ".join(f"{(col[:10] + '...' if len(col) > 10 else col):>15}" for col in group))
        print("-" * 80)
        
        # Print row
        for stat_name, stat_key in zip(stat_names, stat_keys):
            values = [stats[col][stat_key] for col in group]
            print(f"{stat_name:<10} " + " ".join(f"{v:>15.2f}" for v in values))
        
        print('='* 80)

# --------
# bonus function
# --------

def print_dataset_info(df):
    """Print dataset shape and basic info"""
    print("="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Total cells: {df.shape[0] * df.shape[1]}")
    print()
    
    print("Column types:")
    print(f"  Numerical: {len(df.select_dtypes(include=['float64', 'int64']).columns)}")
    print(f"  Categorical: {len(df.select_dtypes(include=['object']).columns)}")
    print()
    
    print("Missing values:")
    total_missing = df.isnull().sum().sum()
    print(f"  Total: {total_missing} ({total_missing / (df.shape[0] * df.shape[1]) * 100:.2f}%)")
    print()
    print("-"*60)

# --------
# main for execution
# --------

def main():
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    exclude_cols = ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    stats = {}
    for col in numerical_cols:
        stats[col] = {
            "count": calculate_count(df[col]),
            "mean": calculate_mean(df[col]),
            "std": calculate_std(df[col]),
            "min": calculate_min(df[col]),
            "p25": calculate_percentile(df[col], 0.25),
            "p50": calculate_percentile(df[col], 0.50),
            "p75": calculate_percentile(df[col], 0.75),
            "max": calculate_max(df[col]),
        }
    
    print_dataset_info(df)
    print_describe_table(stats)

if __name__ == "__main__":
    main()