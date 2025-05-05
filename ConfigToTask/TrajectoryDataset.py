from torch.utils.data import Dataset, DataLoader, random_split

class TrajectoryDataset(Dataset):
    def __init__(self, angles, positions, pressure, seq_len):
        self.angles = angles
        self.positions = positions
        self.pressure = pressure
        self.seq_len = seq_len
        self.samples = []

        # Build sliding window samples
        num_trajectories, traj_len, _ = angles.shape

        for traj_idx in range(num_trajectories):
            for t in range(traj_len - seq_len):

                input_seq = positions[traj_idx, t:t + seq_len, :]
                target_pos = angles[traj_idx, t + seq_len, :]
                self.samples.append((input_seq, target_pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
