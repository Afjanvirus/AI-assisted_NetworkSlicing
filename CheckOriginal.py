import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
import numpy as np
import matplotlib.pyplot as plt

# Define the Network Slice Environment
class NetworkSliceEnv:
    def __init__(self):
        self.current_step = 0
        self.state = self.initial_state()

    def initial_state(self):
        # Static KPIs for both slices
        return {
            'slice1': {'throughput': 6000, 'latency': 20, 'jitter': 5, 'packet_loss': 0.005},
            'slice2': {'throughput': 9000, 'latency': 25, 'jitter': 4, 'packet_loss': 0.003},
            'primSignalStrength': -80, 'recCbc': 2, 'sentCbc': 1,
            'recCbdTime': 1.5, 'recThExceededDelayPkts': 5, 'sentThExceededDelayPkts': 3
        }

    def reset(self):
        self.current_step = 0
        self.state = self.initial_state()
        return self.get_state_vector()

    def get_state_vector(self):
        # Create a 14-element feature vector
        return np.array([
            self.state['primSignalStrength'], self.state['recCbc'], self.state['sentCbc'],
            self.state['recCbdTime'], self.state['recThExceededDelayPkts'], self.state['sentThExceededDelayPkts'],
            self.state['slice1']['throughput'], self.state['slice1']['latency'],
            self.state['slice1']['jitter'], self.state['slice1']['packet_loss'],
            self.state['slice2']['throughput'], self.state['slice2']['latency'],
            self.state['slice2']['jitter'], self.state['slice2']['packet_loss']
        ])

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= 100
        reward = self.compute_reward(action[0], action[1])
        return self.get_state_vector(), reward, done, action

    def compute_reward(self, a1, a2):
        s1, s2 = self.state['slice1'], self.state['slice2']
        throughput = a1 * s1['throughput'] + a2 * s2['throughput']
        latency = a1 * s1['latency'] + a2 * s2['latency']
        jitter = a1 * s1['jitter'] + a2 * s2['jitter']
        loss = a1 * s1['packet_loss'] + a2 * s2['packet_loss']
        reward = throughput - (latency + jitter + loss)
        return reward

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(14, 8)
        self.fc2 = nn.Linear(8, 2)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        alpha = self.softplus(self.fc2(x)) + 1e-6
        return alpha * 10

# Initialize
env = NetworkSliceEnv()
policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=0.01)
gamma = 0.995

rewards_history, loss_history = [], []
allocations_slice1, allocations_slice2 = [], []

# Training Loop
for episode in range(200):
    obs = torch.tensor(env.reset(), dtype=torch.float32)
    done = False
    rewards, actions, states = [], [], []

    while not done:
        alpha = policy(obs)
        dist = Dirichlet(alpha)
        action = dist.sample()
        obs_, reward, done, _ = env.step(action.detach().numpy())
        
        if done:
            print(f"Episode: {episode} Alpha: {alpha} Action: {action}")
        
        rewards.append(reward)
        actions.append(action.detach().numpy())
        states.append(obs)
        obs = torch.tensor(obs_, dtype=torch.float32)

    # Compute returns
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)

    # Policy update
    for state, action, G in zip(states, actions, returns):
        alpha = policy(state)
        dist = Dirichlet(alpha)
        log_prob = dist.log_prob(torch.tensor(action))
        loss = -log_prob * G
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    rewards_history.append(sum(rewards))
    loss_history.append(loss.item())
    allocations_slice1.append(np.mean([a[0] for a in actions]))
    allocations_slice2.append(np.mean([a[1] for a in actions]))

    if episode % 50 == 0:
        print(f"Episode {episode} - Total Reward: {sum(rewards)}, Loss: {loss.item()}")

# Visualization
plt.figure(figsize=(15, 7))

plt.subplot(2, 2, 1)
plt.plot(rewards_history)
plt.title('Total Reward over Episodes')

plt.subplot(2, 2, 2)
plt.plot(loss_history)
plt.title('Loss over Episodes')

plt.subplot(2, 2, 3)
plt.plot(allocations_slice1, label='Slice 1')
plt.plot(allocations_slice2, label='Slice 2')
plt.legend()
plt.title('Resource Allocation Trends')

plt.tight_layout()
plt.show()
