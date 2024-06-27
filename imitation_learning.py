import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DemoDataset(Dataset):
    """Dataset"""
    def __init__(self, input_states, input_actions):
        self.states = input_states
        self.actions = input_actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class MLP(nn.Module):
    """Многослойный перцептрон"""
    def __init__(self, input_size, action_size, neurons=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, action_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the MLP.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, action_size).
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


OBS_SIZE = 17
ACTION_SIZE = 6
NERONS = 401
EPISODES = 10**4
LERN_RATE = 0.01

if __name__=='__main__':

    data = pd.read_csv('HalfCheetah.csv')
    demo_states = data.iloc[:, :OBS_SIZE].values.astype(np.float32)
    demo_actions = data.iloc[:, OBS_SIZE:OBS_SIZE+ACTION_SIZE].values.astype(np.float32)

    dataset = DemoDataset(demo_states, demo_actions)
    dataloader = DataLoader(dataset, batch_size=NERONS, shuffle=True)

    model = MLP(OBS_SIZE, ACTION_SIZE, NERONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LERN_RATE)
    criterion = nn.MSELoss()

    for i in range(EPISODES):
        for states, actions in dataloader:
            states = states.float()
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

        if (i+1) % 1000 == 0:
            print(f'Epoch {i+1}/{EPISODES}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'half_cheetah_im.pth')
