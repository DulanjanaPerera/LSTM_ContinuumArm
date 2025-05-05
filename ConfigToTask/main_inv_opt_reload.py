import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import heapq
import scipy.io
from TrajectoryDataset import TrajectoryDataset
from KinLSTM import KinLSTM
import optuna
from optuna.samplers import TPESampler

USE_FIXED_SEED = True
BASE_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

file_theta = 'C:/Users/dperera/OneDrive - Texas A&M University/TAMU course/NUEN 689 - DL for Engineers/Project/Matlab/Data/original data for training/theta.mat'
file_phi = 'C:/Users/dperera/OneDrive - Texas A&M University/TAMU course/NUEN 689 - DL for Engineers/Project/Matlab/Data/original data for training/phi.mat'
file_p = 'C:/Users/dperera/OneDrive - Texas A&M University/TAMU course/NUEN 689 - DL for Engineers/Project/Matlab/Data/original data for training/p.mat'
file_pressure = 'C:/Users/dperera/OneDrive - Texas A&M University/TAMU course/NUEN 689 - DL for Engineers/Project/Matlab/Data/original data for training/pressure.mat'

# Load the .mat file
data_theta = scipy.io.loadmat(file_theta)
data_phi = scipy.io.loadmat(file_phi)
data_p = scipy.io.loadmat(file_p)
data_pressure = scipy.io.loadmat(file_pressure)

print("Dataset keys:\n")
print(data_theta.keys())
print(data_phi.keys())
print(data_p.keys())
print(data_pressure.keys())

# Extract the data
theta = data_theta['theta_traj']
phi = data_phi['phi_traj']
p = data_p['p']
pressure = data_pressure['pressure_traj']

angles = np.stack((theta, phi), axis=-1)

# Move to the GPU
angles = torch.tensor(angles, dtype=torch.float32)
positions = torch.tensor(p, dtype=torch.float32)
pressure = torch.tensor(pressure, dtype=torch.float32)


# ===================================================
# Generate Dataset
# ===================================================
seq_len = 10

# Create dataset
full_dataset = TrajectoryDataset(angles, positions, pressure, seq_len)

# Split: 70% train, 30% test
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# =====================================================
# 4. TRAINING
# =====================================================

model = KinLSTM(Llayers=3, Lhidden_n=45, fc_hidden=[47, 35, 26]).to(device)
model.load_state_dict(torch.load('best_inv_model.pt'))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001784531637427362)
print(model)

n_epochs = 800

best_models = []
train_rmse_history = []
test_rmse_history = []
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # frequent check for best model


    model.eval()
    with torch.no_grad():
        train_rmse_list = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            rmse = torch.sqrt(criterion(y_pred, y_batch))
            train_rmse_list.append(rmse.item())
        train_rmse = np.mean(train_rmse_list)

        test_rmse_list = []
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            rmse = torch.sqrt(criterion(y_pred, y_batch))
            test_rmse_list.append(rmse.item())
        test_rmse = np.mean(test_rmse_list)

    train_rmse_history.append((epoch, train_rmse))
    test_rmse_history.append((epoch, test_rmse))
    # I only print evalaution result at every 100
    if epoch % 100 == 0:
      print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}")

    if epoch % 50 != 0:
        continue
    # But I select models at every 50
    heapq.heappush(best_models, (test_rmse, epoch, model.state_dict()))

    # I only consider to keep best 5 because best one might be an overfitted model
    if len(best_models) > 5:
        heapq.heappop(best_models)

best_models = sorted(best_models)
best_rmse, best_epoch, best_state_dict = best_models[0]
best_model = KinLSTM(Llayers=3, Lhidden_n=45, fc_hidden=[47, 35, 26]).to(device)
best_model.load_state_dict(best_state_dict)


epochs, train_vals = zip(*train_rmse_history)
_, test_vals = zip(*test_rmse_history)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_vals, label='Train RMSE')
plt.plot(epochs, test_vals, label='Test RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Training and Testing RMSE over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("validation_plot_opt_inv_best.png", dpi=300, bbox_inches='tight')
plt.close()

best_model.eval()
all_targets = []
all_predictions = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = best_model(X_batch)

        all_targets.append(y_batch.cpu().numpy())
        all_predictions.append(y_pred.cpu().numpy())

# Stack all test results
all_targets = np.vstack(all_targets)        # shape (N_test_samples, 3)
all_predictions = np.vstack(all_predictions) # shape (N_test_samples, 3)

print("Shape of the targets and predictions:")
print(all_targets.shape, all_predictions.shape)  # sanity check

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

variables = ['theta', 'phi']

for i in range(2):
    axes[i].plot(all_targets[:100, i], label=f"Actual {variables[i]}", color="blue")
    axes[i].plot(all_predictions[:100, i], label=f"Predicted {variables[i]}", color="red", linestyle='--')
    axes[i].set_title(f"{variables[i].upper()} Prediction")
    axes[i].set_xlabel("Sample")
    axes[i].set_ylabel(variables[i])
    axes[i].legend()

plt.tight_layout()
plt.savefig("prediction_plot_opt_inv_best.png", dpi=300, bbox_inches='tight')
plt.close()

errors = all_predictions - all_targets  # shape (N_test_samples, 3)
abs_errors = np.abs(errors)

# Create error plots
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

for i in range(2):
    axes[i].plot(abs_errors[:100, i], label=f"Absolute Error {variables[i]}", color="green")
    axes[i].set_title(f"{variables[i].upper()} Absolute Error")
    axes[i].set_xlabel("Sample")
    axes[i].set_ylabel("Error")
    axes[i].legend()

plt.tight_layout()
plt.savefig("error_plot_opt_inv_best.png", dpi=300, bbox_inches='tight')
plt.close()