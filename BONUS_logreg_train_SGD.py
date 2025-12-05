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
    # z = w0 + w1*f1 + w2*f2 + w3*f3...
    return sigmoid(np.dot(features, weights) + b) #go to sigmoid to compress the probability

# -----
# Training loop functions
# -----

def compute_loss(y, y_hat):
    m = len(y)
    # add small epsilon to avoid log(0)
    return - (1/m) * np.sum(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))

def update_parameters(w, b, dw, db, learning_rate):
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

def train_one_house_sgd(X, y, house_name, learning_rate=0.01, epochs=1000):
    """
    Stochastic Gradient Descent - updates after each student
    """
    w = np.random.randn(X.shape[1]) * 0.01
    b = 0
    loss_history = []
    
    n_samples = X.shape[0]
    
    for epoch in range(epochs):
        # Shuffle data each epoch (important for SGD!)
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        
        # Loop through EACH student (this is the "stochastic" part)
        for i in range(n_samples):
            # Get ONE student
            x_i = X_shuffled[i]  # Shape: (9,) - one student's features
            y_i = y_shuffled[i]  # Shape: scalar - one label
            
            # Forward pass for this ONE student
            y_hat_i = predict_probability(x_i, w, b)  # Single prediction
            
            # Compute loss for this student
            loss_i = -(y_i * np.log(y_hat_i + 1e-15) + 
                      (1 - y_i) * np.log(1 - y_hat_i + 1e-15))
            epoch_loss += loss_i
            
            # Compute gradients for this ONE student
            error = y_hat_i - y_i
            dw = error * x_i  # Element-wise: (9,) * scalar = (9,)
            db = error
            
            # Update weights immediately (SGD updates after each sample!)
            w, b = update_parameters(w, b, dw, db, learning_rate)
        
        # Average loss for the epoch
        avg_loss = epoch_loss / n_samples
        loss_history.append(avg_loss)
        
        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"{house_name} Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")
    
    return w, b, loss_history


def load_and_preprocess(filename):
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
        print("Usage: python logreg_train.py dataset_train.csv")
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
        w, b, loss_history = train_one_house_sgd(
            X, y_binary, house, 
            learning_rate=0.05,  # You can tune this
            epochs=50
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
        plt.savefig(f'SGD_loss_history_{house}.png')
        plt.close()
        print(f"Please see the loss history @ SGD_loss_history_{house}.png")
    
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