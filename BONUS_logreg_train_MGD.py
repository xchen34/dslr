import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

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

# -----
# The Model - algo
# -----
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_probability(features, weights, b):
    """Calculate probability for one student"""
    return sigmoid(np.dot(features, weights) + b)

# -----
# Training loop functions
# -----

def compute_loss(y, y_hat):
    m = len(y)
    return - (1/m) * np.sum(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))

def compute_gradients(X, y, y_hat):
    m = len(y)
    dw = (1/m) * np.dot(X.T, (y_hat - y))
    db = (1/m) * np.sum(y_hat - y)
    return dw, db

def update_parameters(w, b, dw, db, learning_rate):
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

def train_one_house_MGD(X, y, house_name, learning_rate=0.01, epochs=1000, batch_size = 128):
    w =  np.random.randn(X.shape[1]) * 0.01  # a small random values
    b = 0
    loss_history = []

    for i in range(epochs):
        epoch_loss = 0
        # shuffle data
        permutation = np.random.permutation(len(y))
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for j in range(0, len(y), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            # Forward pass
            y_hat = predict_probability(X_batch, w, b)

            # Compute loss
            loss = compute_loss(y_batch, y_hat)
            loss_history.append(loss)

            # Compute gradients
            dw, db = compute_gradients(X_batch, y_batch, y_hat)

            # Update weights
            w, b = update_parameters(w, b, dw, db, learning_rate)

            batch_loss = compute_loss(y_batch, y_hat)
            epoch_loss += batch_loss
        
        epoch_loss /= (len(y) / batch_size)
        loss_history.append(epoch_loss)
        # (Optional) print progress every N steps
        if i % 100 == 0:
            print(f"{house_name} Epoch {i + 1}|{epochs}: Ave loss = {epoch_loss:.4f}") 
    return w, b, loss_history


def load_and_preprocess(filename):
    """Load data and prepare for training"""
    df = pd.read_csv(filename)
    
    # 1. Extract features and labels
    X = df[selected_features].copy()
    y = df['Hogwarts House']
    
    # 2. Handle missing values
    for col in selected_features:
        mean_val = X[col].mean()
        X[col] = X[col].fillna(mean_val)
    
    # 3. Normalize features (important!)
    normalization_params = {}  # Save these for later!
    for col in selected_features:
        min_val = X[col].min()
        max_val = X[col].max()
        
        # Avoid division by zero
        if max_val - min_val == 0:
            X[col] = 0
        else:
            X[col] = (X[col] - min_val) / (max_val - min_val)
        
        # Save params for test set
        normalization_params[col] = {'min': min_val, 'max': max_val}
    
    # 4. Convert house names to binary labels
    X = X.values
    y = y.values
    
    return X, y, normalization_params

# -----
# Accuracy algo
# -----

def calculate_accuracy(X, y_true, all_weights):
    """
    Calculate prediction accuracy
    X: feature matrix
    y_true: actual house labels (string names)
    all_weights: dictionary with all 4 house models
    """
    predictions = []
    
    for i in range(len(X)):
        student_features = X[i]
        probabilities = {}
        
        # Get probability for each house
        for house in HOUSES:
            w = np.array(all_weights[house]['weights'])
            b = all_weights[house]['bias']
            prob = predict_probability(student_features, w, b)
            probabilities[house] = prob
        
        # Predict the house with highest probability
        predicted_house = max(probabilities, key=probabilities.get)
        predictions.append(predicted_house)
    
    # Calculate accuracy
    predictions = np.array(predictions)
    correct = np.sum(predictions == y_true)
    accuracy = correct / len(y_true)
    
    return accuracy, predictions

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 BONUS_logreg_train_MGD.py <dataset_train.csv>")
        sys.exit(1)

    X, y, normalization_params = load_and_preprocess(sys.argv[1])
    print(f"Loaded {len(X)} students with {len(selected_features)} features")

    # Train one model per house
    all_weights = {}

    for house in HOUSES:
        print(f"\n{'='*50}")
        print(f"Training model for {house}")
        print(f"{'='*50}")
        
        # Create binary labels: 1 if this house, 0 otherwise
        y_binary = (y == house).astype(int)
        
        # Train
        w, b, loss_history = train_one_house_MGD(
            X, y_binary, house, 
            learning_rate=0.05,  # You can tune this
            epochs=1000,
            batch_size = 128
        )
        
        # Save weights for this house
        all_weights[house] = {
            'weights': w.tolist(),  # Convert numpy array to list for JSON
            'bias': float(b),
            'final_loss': loss_history[-1]
        }
        
        print(f"Final loss: {loss_history[-1]:.4f}")

        plt.plot(loss_history)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.savefig(f'MGD_loss_history_{house}.png')
        plt.close()
        print(f"Please see the loss history @ MGD_loss_history_{house}.png")
    
    # Check accuracy
    # Calculate training accuracy
    print(f"\n{'='*50}")
    print("Calculating training accuracy...")
    print(f"{'='*50}")
    
    accuracy, predictions = calculate_accuracy(X, y, all_weights)
    
    print(f"\n✓ Training Accuracy: {accuracy * 100:.2f}%")
    
    # Show per-house breakdown
    print("\nPer-house accuracy:")
    for house in HOUSES:
        house_mask = (y == house)
        house_predictions = predictions[house_mask]
        house_actual = y[house_mask]
        house_accuracy = np.sum(house_predictions == house_actual) / len(house_actual)
        print(f"  {house:12}: {house_accuracy * 100:.2f}% ({np.sum(house_predictions == house_actual)}/{len(house_actual)})")

    # Save everything to file
    output = {
        'features': selected_features,
        'normalization_params': normalization_params,
        'models': all_weights,
        'training_accuracy': float(accuracy)
    }
    
    with open('weights.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*50}")
    print("✓ Training complete!")
    print("✓ Weights saved to weights.json")
    print(f"{'='*50}")
