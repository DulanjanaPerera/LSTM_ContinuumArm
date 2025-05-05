import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io

from TrajectoryDataset import TrajectoryDataset
from KinLSTM import KinLSTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

file_theta = 'C:/Users/dperera/OneDrive - Texas A&M University/TAMU course/NUEN 689 - DL for Engineers/Project/Matlab/Data/trajectory/test7/theta_test_traject.mat'
file_phi = 'C:/Users/dperera/OneDrive - Texas A&M University/TAMU course/NUEN 689 - DL for Engineers/Project/Matlab/Data/trajectory/test7/phi_test_traject.mat'
file_p = 'C:/Users/dperera/OneDrive - Texas A&M University/TAMU course/NUEN 689 - DL for Engineers/Project/Matlab/Data/trajectory/test7/p_test_traject.mat'
file_pressure = 'C:/Users/dperera/OneDrive - Texas A&M University/TAMU course/NUEN 689 - DL for Engineers/Project/Matlab/Data/trajectory/test7/pressure_test_traject.mat'

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

seq_len = 10

# Create dataset
full_dataset = TrajectoryDataset(angles, positions, pressure, seq_len)
test_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)

model = KinLSTM(Llayers=2, Lhidden_n=89, fc_hidden=[79, 111, 101]).to(device)
model.load_state_dict(torch.load('finetune_opr_inv_v10.pt'))
print(model)

model.eval()
all_targets = []
all_predictions = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)

        all_targets.append(y_batch.cpu().numpy())
        all_predictions.append(y_pred.cpu().numpy())

# Stack all test results
all_targets = np.vstack(all_targets)        # shape (N_test_samples, 3)
all_predictions = np.vstack(all_predictions) # shape (N_test_samples, 3)

print("Shape of the targets and predictions:")
print(all_targets.shape, all_predictions.shape)  # sanity check

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

variables = ['P1', 'P2', 'P3']

for i in range(3):
    axes[i].plot(all_targets[:70, i], label=f"Actual {variables[i]}", color="blue")
    axes[i].plot(all_predictions[:70, i], label=f"Predicted {variables[i]}", color="red", linestyle='--')
    axes[i].set_title(f"{variables[i].upper()} Prediction")
    axes[i].set_xlabel("Sample")
    axes[i].set_ylabel(variables[i])
    axes[i].legend()

plt.tight_layout()
plt.savefig("fine_tune_plots/prediction_plot_tracking_ft106.png", dpi=300, bbox_inches='tight')
plt.close()

errors = all_predictions - all_targets  # shape (N_test_samples, 3)
abs_errors = np.abs(errors)

# Create error plots
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

for i in range(3):
    axes[i].plot(abs_errors[:70, i], label=f"Absolute Error {variables[i]}", color="green")
    axes[i].set_title(f"{variables[i].upper()} Absolute Error")
    axes[i].set_xlabel("Sample")
    axes[i].set_ylabel("Error")
    axes[i].legend()

plt.tight_layout()
plt.savefig("fine_tune_plots/error_plot_tracking_ft106.png", dpi=300, bbox_inches='tight')
plt.close()

