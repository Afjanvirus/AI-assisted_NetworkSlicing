import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
import numpy as np
import json
from flask import Flask, request, jsonify

# Define the Network Slice Environment
class NetworkSliceEnv:
    def __init__(self):
        self.current_step = 0

    def initial_state(self, kpis_json):
        """Initialize the environment state using KPIs from JSON."""
        kpis_data = json.loads(kpis_json)
        slice1_kpis = kpis_data['kpis'][0]
        slice2_kpis = kpis_data['kpis'][1]
        
        return {
            'slice1': {
                'throughput': slice1_kpis['throughput'],
                'latency': slice1_kpis['latency'],
                'jitter': slice1_kpis['jitter'],
                'packet_loss': slice1_kpis['packetLoss']
            },
            'slice2': {
                'throughput': slice2_kpis['throughput'],
                'latency': slice2_kpis['latency'],
                'jitter': slice2_kpis['jitter'],
                'packet_loss': slice2_kpis['packetLoss']
            },
            'primSignalStrength': -80,
            'recCbc': 2,
            'sentCbc': 1,
            'recCbdTime': 1.5,
            'recThExceededDelayPkts': 5,
            'sentThExceededDelayPkts': 3
        }

    def reset(self, kpis_json):
        """Reset the environment with the provided KPIs."""
        self.current_step = 0
        self.state = self.initial_state(kpis_json)
        return self.get_state_vector()

    def get_state_vector(self):
        """Convert the state into a vector for the policy network."""
        return np.array([
            self.state['primSignalStrength'],
            self.state['recCbc'],
            self.state['sentCbc'],
            self.state['recCbdTime'],
            self.state['recThExceededDelayPkts'],
            self.state['sentThExceededDelayPkts']
        ])

    def step(self, action):
        """Take a step in the environment based on the action."""
        self.current_step += 1
        done = self.current_step >= 100
        allocation_slice1, allocation_slice2 = action[0], action[1]
        next_state = self.get_state_vector()
        reward = self.compute_reward(allocation_slice1, allocation_slice2)
        return next_state, reward, done, {}

    def compute_reward(self, allocation_slice1, allocation_slice2):
        """Calculate the reward based on resource allocations and KPIs."""
        slice1 = self.state['slice1']
        slice2 = self.state['slice2']
        throughput_reward = allocation_slice1 * slice1['throughput'] + allocation_slice2 * slice2['throughput']
        latency_penalty = allocation_slice1 * slice1['latency'] + allocation_slice2 * slice2['latency']
        jitter_penalty = allocation_slice1 * slice1['jitter'] + allocation_slice2 * slice2['jitter']
        packet_loss_penalty = allocation_slice1 * slice1['packet_loss'] + allocation_slice2 * slice2['packet_loss']
        return throughput_reward - (latency_penalty + jitter_penalty + packet_loss_penalty)

    def save_model(self, policy_network, filepath='trained_model.pth'):
        """Save the trained model's state dictionary."""
        torch.save(policy_network.state_dict(), filepath)
        print(f"Model saved to '{filepath}'")

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 4)  # Input: 6 state features, Output: 4 hidden units
        self.fc2 = nn.Linear(4, 2)  # Output: 2 concentration parameters for Dirichlet
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        alpha = self.softplus(self.fc2(x)) + 1e-6  # Ensure positive values
        return alpha * 10  # Scale for better distribution spread

# Training Function
def train_model(kpis_json):
    """Train the reinforcement learning model using the provided KPIs."""
    env = NetworkSliceEnv()
    policy_network = PolicyNetwork()
    optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
    gamma = 0.995  # Discount factor
    num_episodes = 300

    for episode in range(num_episodes):
        obs = torch.tensor(env.reset(kpis_json), dtype=torch.float32)
        done = False
        rewards = []

        while not done:
            alpha = policy_network(obs)
            dirichlet_dist = Dirichlet(alpha)
            action = dirichlet_dist.sample()
            obs_, reward, done, _ = env.step(action.detach().numpy())
            
            if done:
                print(f"Episode: {episode} Alpha: {alpha} Action: {action}")

            rewards.append(reward)
            obs = torch.tensor(obs_, dtype=torch.float32)

        # Compute discounted returns
        discounted_returns = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_returns.insert(0, cumulative_reward)
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32)

        # Update policy (simplified for clarity)
        for state, reward in zip([obs], rewards):
            alpha = policy_network(state)
            dirichlet_dist = Dirichlet(alpha)
            loss = -dirichlet_dist.log_prob(action) * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    env.save_model(policy_network, 'trained_model.pth')
    return "Training completed and model saved."

# Flask API Setup
app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    """API endpoint to receive KPIs and start training."""
    kpis_json = request.json
    if not kpis_json or 'kpis' not in kpis_json or len(kpis_json['kpis']) != 2:
        return jsonify({"error": "Invalid KPIs JSON format. Expected 'kpis' with two slices."}), 400
    
    result = train_model(json.dumps(kpis_json))
    return jsonify({"message": result})

if __name__ == '__main__':
    app.run(debug=True)