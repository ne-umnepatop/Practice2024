import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class HalfCheetahDataset(Dataset):
    """Dataset"""
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class MLP(nn.Module):
    """Многослойный перцептрон"""
    def __init__(self, input_size, ACTION_SIZE, NERONS=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, NERONS)
        self.fc2 = nn.Linear(NERONS, NERONS)
        self.fc3 = nn.Linear(NERONS, ACTION_SIZE)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


OBS_SIZE = 16
ACTION_SIZE = 6
NERONS = 64
EPISODES = 100
LERN_RATE = 0.001

if __name__=='__main__':

    data = pd.read_csv('HalfCheetah.csv')
    demo_states = data.iloc[:, :OBS_SIZE].values
    demo_actions = data.iloc[:, OBS_SIZE:OBS_SIZE+ACTION_SIZE].values

    dataset = HalfCheetahDataset(demo_states, demo_actions)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MLP(OBS_SIZE, ACTION_SIZE, NERONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LERN_RATE)
    criterion = nn.MSELoss()

    for i in range(EPISODES):
        for states, actions in dataloader:
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{i+1}/{EPISODES}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'half_cheetah_im.pth')
