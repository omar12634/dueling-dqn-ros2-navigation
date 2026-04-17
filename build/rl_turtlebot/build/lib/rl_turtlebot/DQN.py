import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.model = DQN(state_size, action_size)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01


    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        state = torch.FloatTensor(state)
        q_values = self.model(state)

        return torch.argmax(q_values).item()
