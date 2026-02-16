from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define the Network Slice Environment
class NetworkSliceEnv:
    def __init__(self):
        self.current_step = 0
        self.state = self.initial_state()
        self.last_action = None  # To store the last action taken

    def initial_state(self):
        return {
            'slice1': {'throughput': 200, 'latency': 20, 'jitter': 5, 'packet_loss': 0.005},
            'slice2': {'throughput': 200, 'latency': 25, 'jitter': 4, 'packet_loss': 0.003},
            'primSignalStrength': -80, 'recCbc': 2, 'sentCbc': 1,
            'recCbdTime': 1.5, 'recThExceededDelayPkts': 5, 'sentThExceededDelayPkts': 3
        }

    def reset(self):
        self.current_step = 0
        self.state = self.initial_state()
        self.last_action = None  # Reset last action
        return self.get_state_vector()

    def get_state_vector(self):
        state_signals = [
            self.state['primSignalStrength'], self.state['recCbc'], self.state['sentCbc'],
            self.state['recCbdTime'], self.state['recThExceededDelayPkts'], self.state['sentThExceededDelayPkts'],
            self.state['slice1']['throughput'], self.state['slice1']['latency'],
            self.state['slice1']['jitter'], self.state['slice1']['packet_loss'],
            self.state['slice2']['throughput'], self.state['slice2']['latency'],
            self.state['slice2']['jitter'], self.state['slice2']['packet_loss']
        ]
        return np.array(state_signals)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= 100
        self.last_action = action  # Store the last action taken
        allocation_slice1, allocation_slice2 = action[0], action[1]
        next_state = self.get_state_vector()
        reward = self.compute_reward(allocation_slice1, allocation_slice2)
        return next_state, reward, done, action

    def compute_reward(self, allocation_slice1, allocation_slice2):
        slice1, slice2 = self.state['slice1'], self.state['slice2']
        throughput_reward = allocation_slice1 * slice1['throughput'] + allocation_slice2 * slice2['throughput']
        latency_penalty = allocation_slice1 * slice1['latency'] + allocation_slice2 * slice2['latency']
        jitter_penalty = allocation_slice1 * slice1['jitter'] + allocation_slice2 * slice2['jitter']
        packet_loss_penalty = allocation_slice1 * slice1['packet_loss'] + allocation_slice2 * slice2['packet_loss']
        reward = throughput_reward - (latency_penalty + jitter_penalty + packet_loss_penalty)
        return reward

# Neural Network for predicting Dirichlet distribution parameters
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(14, 8)
        self.fc2 = nn.Linear(8, 2)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        alpha = self.softplus(self.fc2(x)) + 1e-6
        alpha = torch.clamp(alpha, min=0.6, max=6)
        alpha = alpha + 1e-6
        return alpha #* 10

# Initialize environment and policy
env = NetworkSliceEnv()
policy_network = PolicyNetwork()
optimizer = optim.Adam(policy_network.parameters(), lr=0.01)
gamma = 0.995  # Discount factor

# Storage for visualization
rewards_history, loss_history, allocations_slice1, allocations_slice2 = [], [], [], []
running_baseline = 0
alpha_baseline = 0.9
num_episodes = 20

@app.route('/')
def home():
    return "Welcome to the AI-assisted Network Slicing API!"

@app.route('/get_state', methods=['POST'])
def get_state():
    print("Received a request to /get_state")
    data = request.json
    print("Data received:", data)

    # Check if the incoming data is structured correctly
    if 'kpis' not in data or 'networkState' not in data or len(data['kpis']) != 2:
        return jsonify({"error": "Invalid data format!"}), 400

    # Update environment state with new KPIs and network state
    kpis = data['kpis']
    network_state = data['networkState']

    # KPIs for Slice 1
    env.state['slice1']['throughput'] = kpis[0]['throughput']
    env.state['slice1']['latency'] = kpis[0]['latency']
    env.state['slice1']['jitter'] = kpis[0]['jitter']
    env.state['slice1']['packet_loss'] = kpis[0]['packetLoss']

    # KPIs for Slice 2
    env.state['slice2']['throughput'] = kpis[1]['throughput']
    env.state['slice2']['latency'] = kpis[1]['latency']
    env.state['slice2']['jitter'] = kpis[1]['jitter']
    env.state['slice2']['packet_loss'] = kpis[1]['packetLoss']

    # Update the network state
    env.state['primSignalStrength'] = network_state['signalStrength']
    env.state['recCbc'] = network_state['recCbc']
    env.state['sentCbc'] = network_state['sentCbC']
    env.state['recCbdTime'] = network_state['recCbdTime']
    env.state['recThExceededDelayPkts'] = network_state['recThExceededDelayPkts']
    env.state['sentThExceededDelayPkts'] = network_state['sentThExceededDelayPkts']

    # Print the new KPIs received
    print("New KPIs and network state received:")
    print(env.state)

    # Optionally retrain the model here
    retrain_model()

    return jsonify({"message": "State updated successfully!"})

def retrain_model():
    for episode in range(num_episodes):
        obs = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        log_probs, rewards, actions, states = [], [], [], []

        while not done:
            alpha = policy_network(obs)
            dirichlet_dist = Dirichlet(alpha)
            action = dirichlet_dist.sample()
            obs_, reward, done, _ = env.step(action.detach().numpy())

            # Print the episode number, alpha values, and the sampled action
            print(f"Episode: {episode + 1} Alpha: {alpha} Action: {action}")

            rewards.append(reward)
            actions.append(action.detach().numpy())
            states.append(obs)

            obs = torch.tensor(obs_, dtype=torch.float32)

        discounted_returns = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_returns.insert(0, cumulative_reward)
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32)

        advantages = discounted_returns

        for state, action, G in zip(states, actions, rewards):
            alpha = policy_network(state)
            dirichlet_dist = Dirichlet(alpha)
            log_prob = dirichlet_dist.log_prob(torch.tensor(action))
            entropy_loss = -dirichlet_dist.entropy().mean()
            loss = -log_prob * G
            # Total loss with entropy regularization
            total_loss = loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

@app.route('/allocate_resource', methods=['GET'])
def allocate_resource():
    # Return the last action taken
    if env.last_action is not None:
        return jsonify({
            "allocation": {
                "slice1": env.last_action[0].item(),
                "slice2": env.last_action[1].item()
            }
        })
    else:
        return jsonify({"message": "No action has been taken yet."}), 404

if __name__ == '__main__':
    app.run(debug=True)
