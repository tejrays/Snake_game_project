import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class LinearQNet(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, folder='model', filename='model.pth'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, filename)
        torch.save(self.state_dict(), path)


class QTrainer:
    def __init__(self, model, lr=0.001, gamma=0.9):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # convert to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if state.ndim == 1:
            # add batch dim
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            # action is one-hot or index
            if action.ndim == 2:  # one-hot
                act_idx = torch.argmax(action[idx]).item()
            else:
                act_idx = action[idx].item()
            target[idx][act_idx] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
