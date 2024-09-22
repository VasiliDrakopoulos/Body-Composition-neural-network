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

# Data for Females (Weight, Age, Fat, Lean)
data_females = np.array([
    [21.65, 103, 2.28, 17.0],
    [21.31, 103, 2.38, 15.85],
    [20.52, 103, 3.01, 15.5],
    [22.07, 103, 2.29, 16.36],
    [20.84, 103, 1.11, 17.2],
    [22.46, 107, 3.78, 14.34],
    [22.9, 159, 3.18, 18.02],
    [23.75, 159, 3.68, 17.61],
    [22.18, 159, 3.45, 16.8],
    [22.86, 159, 3.08, 18.1],
    [24.43, 159, 3.35, 18.86],
    [25.37, 163, 4.81, 19.38],
    [20.5, 62, 2.31, 16.62],
    [21, 68, 2.47, 16.82],
    [19.5, 62, 1.95, 15.83],
    [20, 68, 2.09, 16.19],
    [19.5, 62, 2.64, 15.59],
    [20.5, 68, 2.67, 16.24],
    [20.5, 62, 1.68, 17.06],
    [21, 68, 1.76, 17.5],
    [19, 55, 1.65, 15.53],
    [19.5, 61, 1.89, 16.73],
    [18.5, 55, 2.45, 14.74],
    [19.5, 61, 2.11, 16.41],
    [18, 55, 1.82, 15.08],
    [19.5, 61, 2.66, 15.4],
    [19, 55, 2.63, 14.23],
    [20, 61, 1.71, 16.94],
    [26.5, 154, 5.58238268,	16.1828613],
    [26.7, 154, 4.43882132, 18.8140011],
    [32.7, 154,	9.25996113,	20.3085938],
    [30.9, 154,	5.46777964,	20.5322094],
    [48.9, 154,	15.7847214,	25.0771923],
    [31.8, 154,	5.80681562,	21.8518982],
    [30.7, 154,	4.14981222,	23.0194912],
    [32.7, 154,	5.36434555,	22.5599442],
    [29.3, 154,	5.13636112,	20.9770203],
    [32.3, 154,	4.63249731,	23.9652176],
    [27.6, 154,	2.79675388,	20.6412983],
    [32.1, 154,	8.02087212,	20.853672],
    [44.1, 154,	15.0796051,	24.5547314],
    [23.28, 122, 2.73, 18.79],
    [22.46, 122, 3.04, 17.74],
    [27.71, 122, 6.77, 18.82],
    [22.46, 123, 2.82, 18.46],
    [20.81, 123, 2.02, 17.33],
    [21.78, 123, 2.13, 17.84],
    [22.09, 134, 2.43, 18.20],
    [21.54, 134, 2.57, 17.29],
    [21.82, 134, 1.85, 18.11],
    [23.33, 134, 2.33, 19.29],
    [19.93, 135, 1.38, 15.73],
    [21.74, 135, 1.83, 17.05],
    [22.96, 144, 1.74, 19.67],
    [22.28, 144, 2.55, 18.09],
    [22.89, 147, 1.22, 19.37],
    [17.87, 147, 0.48, 15.50],
    [31.77, 252, 4.17, 25.78],
    [27.34, 252, 5.76, 20.02],
    [26, 252, 4.16,	18.47],
    [29.07, 252, 7.65, 19.75],
    [26.82, 230, 4.47, 19.24],
    [23.88, 227, 4.03, 16.46],
    [27.8, 223, 5.37, 18.82],
    [23.73, 214, 3.72, 19.04],
    [24.62, 214, 3.72, 19.33],
    [22.83, 214, 3.58, 18.5],
    [24.1, 214, 3.81, 19.14],
    [26.77, 214, 4.61, 21.11],
    [27.95, 218, 6.85, 19.35],
    [20.85, 141, 1.72, 17.61],
    [18.06, 141, 1.87, 14.81],
    [19.43, 139, 2.53, 15.43],
    [19.26, 139, 2.44, 15.43],
    [18.04, 139, 1.59, 15.06],
    [20.54, 139, 1.68, 17.41],
    [20.77, 138, 2.12, 16.96],
    [19.31, 138, 1.72, 16.2],
    [18.97, 57, 1.84, 15.05],
    [22.03, 57, 2.8, 17.53],
    [21.37, 57, 2.49, 16.93],
    [26.49, 57, 2.05, 22.65],
    [17.7, 56, 1.81, 15.645],
    [18.15, 56, 1.846667, 14.78],
    [18.25, 56, 1.99, 15.75],
    [17.91, 56, 2.053333, 15.54333],
    [18.41, 56, 1.415, 15.16],
    [18.44, 56, 1.62, 15.62],
    [19.01, 56, 1.54, 15.575],
    [18.65, 56, 1.905, 15.285],
    [20.4, 84, 2.92, 18.2],
    [19.46, 84, 2.22, 17.06],
    [21.25, 84, 1.605, 17.205],
    [19.85, 84, 2.21, 16.32],
    [20.37, 84, 2.135, 16.805],
    [19.82, 84, 2.075, 17.11],
    [22.15, 84, 1.99, 17.11],
    [20.59, 84, 2.125, 17.435],
    [21.48, 126, 3.006667, 18.07],
    [20.54, 126, 2.766667, 17.98],
    [21.7, 126, 2.01, 18.065],
    [21.21, 126, 3.25, 16.915],
    [22.29, 126, 2.796667, 18.65],
    [21.65, 126, 3.115, 17.06],
    [22.14, 126, 2.59, 16.945],
    [22.53, 126, 2.85, 17.73333]])

# Features: weight, age
X = data_females[:, :2]
# Targets: fat, lean
Y = data_females[:, 2:4]

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
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
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

    for epoch in range(2000):
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
