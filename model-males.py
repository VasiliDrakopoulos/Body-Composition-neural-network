import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd

# Data for Males (Weight, Age, Fat, Lean)
data_males = np.array([
    [32.55, 99, 3.28, 25.93],
    [31.29, 99, 2.82, 25.59],
    [31.28, 99, 4.56, 22.96],
    [32.17, 100, 2.21, 26.57],
    [30.92, 100, 3.33, 25.64],
    [31.25, 100, 3.66, 25.46],
    [36.41, 155, 7.46, 26.13],
    [36.41, 155, 6.14, 25.09],
    [37.55, 155, 8.44, 26.02],
    [35.92, 156, 3.27, 30.53],
    [33.45, 156, 4.42, 25.07],
    [32.52, 156, 4.21, 26.11],
    [22.5, 61, 1.66, 19.36],
    [23.5, 67, 2.09, 20.22],
    [21.5, 61, 2.53, 18.1],
    [23.5, 67, 2.61, 18.94],
    [22.5, 61, 1.3, 19.96],
    [23, 67, 2.49, 19.5],
    [21.5, 62, 2.19, 17.93],
    [23, 68, 2.66, 19.02],
    [23.5, 58, 1.96, 20.19],
    [24, 64, 2.4, 19.97],
    [23, 58, 1.75, 20.47],
    [23.5, 64, 2.05, 20.15],
    [34.7, 154, 8.70228004, 21.5518398],
    [38.5, 154, 9.46806622, 24.5032272],
    [43.1, 154, 11.7377081, 26.1736107],
    [45.2, 154, 9.79034996, 26.9695377],
    [50.9, 154, 15.4204025, 28.9130821],
    [35.7, 154, 7.50866747, 23.9101009],
    [36.1, 154, 6.91895819, 25.6537952],
    [37.2, 154, 9.19349098, 24.2069225],
    [38.8, 154, 10.1406164, 23.9002628],
    [32.8, 154, 5.22834539, 23.5811596],
    [40.9, 154, 10.2703247, 25.060154],
    [41.3, 154, 10.6309223, 25.6284294],
    [39.5, 154, 9.63011646, 25.1363754],
    [33.30, 124, 5.27, 26.39],
    [30.54, 124, 3.33, 25.22],
    [31.78, 124, 3.52, 26.06],
    [33.49, 121, 6.50, 25.28],
    [34.40, 121, 7.53, 25.45],
    [36.15, 121, 7.57, 25.69],
    [30.32, 134, 0.33, 26.91],
    [32.53, 134, 1.46, 28.50],
    [30.12, 134, 1.60, 25.44],
    [32.34, 134, 1.37, 27.09],
    [25.21, 134, 2.65, 20.56],
    [29.52, 134, 3.35, 23.82],
    [28.15, 151, 2.53, 23.85],
    [27.50, 151, 3.29, 21.96],
    [30.06, 148, 3.91, 24.16],
    [32.42, 148, 5.43, 24.72],
    [33.87, 252, 4.65, 27.09],
    [30.03, 252, 3.73, 23.34],
    [28.12, 223, 2.93, 21.79],
    [26.86, 223, 3, 20.58],
    [29.58, 223, 4.35, 21.79],
    [31.25, 223, 4.75, 22.48],
    [30.52, 223, 4, 23.01],
    [38.45, 210, 9.47, 27.49],
    [38.46, 210, 10.39, 26.99],
    [36.26, 210, 7.67, 27.4],
    [35.8, 211, 4.33, 30.2],
    [33.65, 211, 5.11, 27.08],
    [32.8, 211, 5.02, 26.05],
    [24.73, 139, 2.29, 20.53],
    [25.49, 139, 2.06, 21.78],
    [25.42, 136, 1.83, 24],
    [26.23, 136, 1.89, 22.5],
    [23.14, 58, 1.71, 20.17],
    [23.4, 57, 2.42, 19.65],
    [25.43, 57, 2.69, 21.18],
    [27.83, 56, 1.796667, 18.76333],
    [26.11, 56, 2.406667, 23.66333],
    [28.46, 56, 2.113333, 24.68],
    [23.54, 56, 2.17, 25.715],
    [26.58, 56, 2.056667, 20.89],
    [23.95, 56, 2.135, 23.745],
    [23.62, 56, 2.09, 20.97],
    [21.58, 56, 1.866667, 22.24333],
    [29.37, 84, 3.283333, 21.12],
    [28.29, 84, 2.73, 25.75],
    [31.3, 84, 3.03, 27.37],
    [27.93, 84, 2.87, 24.21],
    [30.1, 84, 3.37, 25.375],
    [27.69, 84, 3.025, 23.145],
    [27.45, 84, 2.9, 24.46],
    [24.95, 84, 2.46, 23.3],
    [30.99, 126, 4.605, 22.445],
    [30.63, 126, 3.926667, 25.68],
    [34.59, 126, 4.863333, 27.94],
    [30.19, 126, 3.093333, 26.05],
    [33.41, 126, 3.796667, 24.99333],
    [29.7, 126, 5.633333, 25.96],
    [28.62, 126, 3.5, 24.26],
    [28.4, 126, 3.42, 24.93],
    [23.40, 42, 0.80, 21.43],
    [24.10, 42, 0.93, 20.93],
    [24.70, 42, 1.37, 21.30],
    [26.40, 42, 1.78, 22.91],
    [25.70, 42, 0.41, 23.08],
    [25.50, 42, 0.83, 23.18],
    [23.10, 42, 1.05, 20.80],
    [22.20, 42, 0.83, 19.99],
    [22.60, 42, 0.22, 20.63],
    [24.50, 42, 0.63, 22.96],
    [25.10, 42, 0.41, 22.95],
    [27.30, 70, 1.60, 24.59],
    [27.20, 70, 1.96, 24.10],
    [27.80, 70, 3.61, 23.20],
    [30.30, 70, 2.44, 26.80],
    [28.80, 70, 0.81, 26.12],
    [28.90, 70, 2.36, 26.35],
    [28.40, 70, 1.86, 25.33],
    [25.00, 70, 1.88, 22.06],
    [28.00, 70, 2.83, 24.88],
    [26.90, 70, 3.91, 23.94],
    [27.90, 70, 2.48, 24.43],
    [29.00, 98, 1.92, 24.77],
    [30.90, 98, 4.02, 25.75],
    [31.00, 98, 4.04, 25.34],
    [31.30, 98, 2.61, 27.79],
    [31.50, 98, 0.54, 29.42],
    [31.40, 98, 3.25, 29.50],
    [31.10, 98, 2.62, 26.92],
    [27.50, 98, 2.27, 26.79],
    [30.10, 98, 2.70, 24.39],
    [30.00, 98, 4.09, 25.70],
    [32.00, 98, 2.50, 26.02],
    [31.80, 154, 2.43, 25.69],
    [35.90, 154, 4.74, 26.19],
    [35.50, 154, 5.11, 26.05],
    [39.20, 154, 6.09, 29.59],
    [33.00, 154, 1.01, 30.83],
    [34.40, 154, 5.06, 30.94],
    [34.90, 154, 3.47, 28.90],
    [30.20, 154, 3.12, 25.77],
    [32.90, 154, 6.29, 28.11],
    [32.50, 154, 6.27, 26.43],
    [33.30, 154, 3.94, 26.40]])

# Features: weight, age
X = data_males[:, :2]
# Targets: fat, lean
Y = data_males[:, 2:4]

# Split data into training (80%) and testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Data normalization
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(2, 256)
        self.bn1 = nn.BatchNorm1d(256) 
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)  
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64) 
        self.output = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.2)  # Regularization
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))  
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.layer2(x)))  
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.layer3(x)))  
        x = self.output(x) 
        return x

def train_model(X_train, Y_train, X_test, Y_test):
    model = NeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_list = []  # To store the training loss values
    val_loss_list = []  # To store the validation loss values
    best_val_loss = float('inf')
    patience = 50
    trigger_times = 0

    for epoch in range(4000):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, Y_test)
        
        loss_list.append(loss.item())
        val_loss_list.append(val_loss.item())
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping at epoch:', epoch)
                break
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
    
    plt.plot(loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    return model

def plot_predictions(X_train, Y_train, X_test, Y_test, model):
    model.eval()
    with torch.no_grad():
        train_predictions = scaler_Y.inverse_transform(model(X_train_tensor).numpy())
        test_predictions = scaler_Y.inverse_transform(model(X_test_tensor).numpy())
    
    # Calculate R-squared values for training and testing data separately
    r2_train_fat = r2_score(Y_train[:, 0], train_predictions[:, 0])
    r2_train_lean = r2_score(Y_train[:, 1], train_predictions[:, 1])
    r2_test_fat = r2_score(Y_test[:, 0], test_predictions[:, 0])
    r2_test_lean = r2_score(Y_test[:, 1], test_predictions[:, 1])

    print(f'R-squared for training data (Fat): {r2_train_fat:.4f}')
    print(f'R-squared for training data (Lean): {r2_train_lean:.4f}')
    print(f'R-squared for test data (Fat): {r2_test_fat:.4f}')
    print(f'R-squared for test data (Lean): {r2_test_lean:.4f}')

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Training data
    plt.subplot(2, 2, 1)
    plt.scatter(Y_train[:, 0], train_predictions[:, 0], c="blue", s=30, label="Training data (Fat)")
    plt.plot(Y_train[:, 0], Y_train[:, 0], 'k--', lw=2, label='Perfect Fit')
    plt.xlabel('Actual Fat')
    plt.ylabel('Predicted Fat')
    plt.title('Training Data - Fat')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.scatter(Y_train[:, 1], train_predictions[:, 1], c="cyan", s=30, label="Training data (Lean)")
    plt.plot(Y_train[:, 1], Y_train[:, 1], 'k--', lw=2, label='Perfect Fit')
    plt.xlabel('Actual Lean')
    plt.ylabel('Predicted Lean')
    plt.title('Training Data - Lean')
    plt.legend()

    # Test data
    plt.subplot(2, 2, 3)
    plt.scatter(Y_test[:, 0], test_predictions[:, 0], c="red", s=30, label="Test data (Fat)")
    plt.plot(Y_test[:, 0], Y_test[:, 0], 'k--', lw=2, label='Perfect Fit')
    plt.xlabel('Actual Fat')
    plt.ylabel('Predicted Fat')
    plt.title('Test Data - Fat')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.scatter(Y_test[:, 1], test_predictions[:, 1], c="magenta", s=30, label="Test data (Lean)")
    plt.plot(Y_test[:, 1], Y_test[:, 1], 'k--', lw=2, label='Perfect Fit')
    plt.xlabel('Actual Lean')
    plt.ylabel('Predicted Lean')
    plt.title('Test Data - Lean')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model

model = train_model(X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor)

# Plot the predictions and R-squared values
plot_predictions(X_train_scaled, Y_train, X_test_scaled, Y_test, model)

def predict_fat_lean(weights, ages):
    model.eval()
    inputs = np.column_stack((weights, ages))
    inputs_scaled = scaler_X.transform(inputs)
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs_tensor)
    predictions_scaled = outputs.numpy()
    predictions = scaler_Y.inverse_transform(predictions_scaled)
    return predictions

def export_to_excel(filepath, weights, ages, predictions):
    df = pd.DataFrame({
        'Weight': weights,
        'Age': ages,
        'Predicted Fat': predictions[:, 0],
        'Predicted Lean': predictions[:, 1]
    })
    df.to_excel(filepath, index=False)
    messagebox.showinfo("Export Successful", f"Predictions have been exported to {filepath}")

def on_predict():
    input_data = text_box.get("1.0", tk.END).strip().split('\n')
    weights = []
    ages = []
    for line in input_data:
        try:
            weight, age = map(float, line.split())
            weights.append(weight)
            ages.append(age)
        except ValueError:
            messagebox.showerror("Input Error", f"Invalid input: {line}")
            return

    weights = np.array(weights)
    ages = np.array(ages)
    predictions = predict_fat_lean(weights, ages)


    result_text = "Weight | Age | Predicted Fat | Predicted Lean\n"
    result_text += "\n".join([f"{w} | {a} | {f:.2f} | {l:.2f}" for w, a, f, l in zip(weights, ages, predictions[:, 0], predictions[:, 1])])
    messagebox.showinfo("Predictions", result_text)

    filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if filepath:
        export_to_excel(filepath, weights, ages, predictions)

# GUI Setup
root = tk.Tk()
root.title("Fat and Lean Prediction")

tk.Label(root, text="Enter Weight and Age pairs (one pair per line, separated by space):").pack()
text_box = tk.Text(root, height=10, width=30)
text_box.pack()

predict_button = tk.Button(root, text="Predict", command=on_predict)
predict_button.pack()

root.mainloop()
