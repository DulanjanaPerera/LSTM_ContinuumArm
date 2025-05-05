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

file_theta = 'C:/Users/dperera/OneDrive - Texas A&M University/TAMU course/NUEN 689 - DL for Engineers/Project/data/theta.mat'
file_phi = 'C:/Users/dperera/OneDrive - Texas A&M University/TAMU course/NUEN 689 - DL for Engineers/Project/data/phi.mat'
file_p = 'C:/Users/dperera/OneDrive - Texas A&M University/TAMU course/NUEN 689 - DL for Engineers/Project/data/p.mat'
file_pressure = 'C:/Users/dperera/OneDrive - Texas A&M University/TAMU course/NUEN 689 - DL for Engineers/Project/data/pressure.mat'

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
def objective(trial):
    if USE_FIXED_SEED:
        torch.manual_seed(BASE_SEED)
        np.random.seed(BASE_SEED)
    else:
        seed = BASE_SEED + trial.number
        torch.manual_seed(seed)
        np.random.seed(seed)

    lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
    lstm_hidden = trial.suggest_int('lstm_hidden', 32, 128)
    n_fc = trial.suggest_int('n_fc', 1, 3)
    fc_hidden = [trial.suggest_int(f'fc_hidden_{i}', 16, 64) for i in range(n_fc)]
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

    model = KinLSTM(Llayers=lstm_layers, Lhidden_n=lstm_hidden, fc_hidden=fc_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_models = []
    for epoch in range(200):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    test_rmse_list = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            rmse = torch.sqrt(criterion(y_pred, y_batch))
            test_rmse_list.append(rmse.item())
    mean_rmse = np.mean(test_rmse_list)
    trial.set_user_attr("model_state_dict", model.state_dict())
    return mean_rmse

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=30))
study.optimize(objective, n_trials=40)

print("Best trial:")
print(study.best_trial.params)

# Rebuild and save best model
best_params = study.best_trial.params
best_model = KinLSTM(
    Llayers=best_params['lstm_layers'],
    Lhidden_n=best_params['lstm_hidden'],
    fc_hidden=[best_params[f'fc_hidden_{i}'] for i in range(best_params['n_fc'])]
).to(device)
best_model.load_state_dict(study.best_trial.user_attrs["model_state_dict"])
torch.save(best_model.state_dict(), "best_inv_model.pt")

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
plt.show()

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
plt.show()