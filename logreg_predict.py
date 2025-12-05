import sys
import json
import pandas as pd
import numpy as np

selected_features = [
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

HOUSES = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_one_vs_all(X, weights_dict):
    predictions = []
    for student in X:
        house_probs = {}
        for house, params in weights_dict.items():
            w, b = params["weights"], params["bias"]
            z = np.dot(student, w) + b
            house_probs[house] = sigmoid(z)
            # Optional: for viewing all 4 classifier results
            # print (f"{house:12} : {house_probs[house]* 100:.2f}%")
        
        # pick the house with the highest probability
        predicted_house = max(house_probs, key=house_probs.get)
        predictions.append(predicted_house)
    return predictions

def normalize_features(X, normalization_params):
    """
    Normalize test data using training data parameters
    """
    X_normalized = X.copy()
    
    for col in selected_features:
        min_val = normalization_params[col]['min']
        max_val = normalization_params[col]['max']
        
        if max_val - min_val == 0:
            X_normalized[col] = 0
        else:
            X_normalized[col] = (X_normalized[col] - min_val) / (max_val - min_val)
    
    return X_normalized

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 logreg_predict.py <dataset_test.csv> <weights.json>")
        sys.exit(1)
    
    test_file = sys.argv[1]
    weights_file = 'weights.json'

    print(f"Loading test data from {test_file}...")
    df = pd.read_csv(test_file)
    print(f"Loaded {len(df)} students")

    print(f"Loading weights from {weights_file}...")
    with open(weights_file, 'r') as f:
        weights_data = json.load(f)
    
    # 1. Extract features and labels
    X = df[selected_features].copy()

    # 2. Handle missing values
    for col in selected_features:
        mean_val = X[col].mean()
        X[col] = X[col].fillna(mean_val)
    
    # 3. Normalize features (important!)
    print("Normalizing features...")
    X_normalized = normalize_features(X, weights_data['normalization_params'])
    
    # 4. Predict with model from json
    print("Making predictions...")
    predictions = predict_one_vs_all(X_normalized.values, weights_data['models'])

    # 5. Create output dataframe
    output_df = pd.DataFrame({
        'Index': range(len(predictions)),
        'Hogwarts House': predictions
    })

    output_file = 'houses.csv'
    output_df.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to {output_file}")
    
    # Optional: Show distribution
    print(f"\nPrediction distribution of {len(predictions)} students:")
    for house in HOUSES:
        count = predictions.count(house)
        print(f"  {house:12}: {count:3} students ({count/len(predictions)*100:.1f}%)")

