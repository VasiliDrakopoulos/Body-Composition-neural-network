# Body Composition Neural Network

This repository contains a PyTorch-based neural network model designed to predict fat mass and lean mass in chow-fed mice based on their bodyweight and age in days.

## Project Overview

The neural network model is trained on a dataset of chow-fed male or female mice, with input features being their **bodyweight** and **age**, and the target outputs being their **fat mass** and **lean mass**.

The model architecture consists of:
- Fully connected layers
- Batch normalization
- Dropout for regularization

The training process utilizes **mean squared error (MSE)** as the loss function and **Adam** optimizer for updating weights.

## Key Features
- **Training and Testing**: The model is trained on 80% of the data and tested on the remaining 20%.
- **R-squared Evaluation**: R-squared scores are calculated for both training and testing datasets.
- **Predictions**: Users can input mouse bodyweights and ages, and the model will predict the fat and lean mass.
- **Graphical Interface**: A Tkinter GUI is included for user input and exporting results to Excel.
- **Early Stopping**: Training automatically halts when the validation loss stops improving.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/VasiliDrakopoulos/body-composition-neural-network.git
    cd body-composition-neural-network
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the GUI:
    ```bash
    python app.py
    ```

## Usage

This model can be used by researchers as control data for fat and lean mass to predict changes from an expected norm.

### Model Training and Evaluation

The script will automatically train the model for up to 4000 epochs or early stop if validation loss plateaus. Once the model is trained, it will output training and validation losses and plot them over the epochs.

### Prediction Example

Users can input mouse bodyweight and age into the GUI or the Python script to get predictions for fat and lean mass. The results can be saved in Excel format.





