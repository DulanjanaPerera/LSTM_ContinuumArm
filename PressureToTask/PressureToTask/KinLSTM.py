import torch.nn as nn

# class KinLSTM(nn.Module):
#     def __init__(self, Llayers=1, Lhidden_n=50, Nlayers=1, nhidden=[50]):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=3, hidden_size=Lhidden_n, num_layers=Llayers, batch_first=True)
#         self.linear = nn.Linear(Lhidden_n, 3)
#         # layers = []
#         # for i in range(len(nhidden) - 1):
#         #     layers.append(nn.Linear(nhidden[i], nhidden[i + 1]))
#         #     layers.append(nn.ReLU())
#
#         # self.fc = nn.Sequential(*layers)
#
#     def forward(self, x):
#         x, _ = self.lstm(x)
#         x = self.linear(x[:,-1,:])
#         return x

import torch.nn as nn

class KinLSTM(nn.Module):
    def __init__(self, Llayers=1, Lhidden_n=50, fc_hidden=[]):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=Lhidden_n, num_layers=Llayers, batch_first=True)

        layers = []
        input_dim = Lhidden_n
        for h in fc_hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 3))  # Output layer
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Use the last timestep
        x = self.fc(x)
        return x
