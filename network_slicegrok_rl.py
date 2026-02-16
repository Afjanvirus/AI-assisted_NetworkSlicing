import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
import numpy as np
import matplotlib.pyplot as plt
import requests
from typing import Dict, List, Tuple
from flask import Flask, request, jsonify
import threading
import time

# Global variable to store the latest KPIs received by the Flask server
latest_kpis = None

# Flask application
app = Flask(__name__)

@app.route('/kpis', methods=['POST'])
def receive_kpis():
    global latest_kpis
    data = request.get_json()
    if not data or 'kpis' not in data:
        return jsonify({"error": "Invalid JSON format. Expected 'kpis' key."}), 400
    latest_kpis = data
    return jsonify({"message": "KPIs received successfully", "kpis": latest_kpis}), 200

# Function to fetch network data (modified to use Flask-stored KPIs)
def fetch_network_data() -> Dict:
    global latest_kpis
    if latest_kpis is not None:
        return latest_kpis
    else:
        print("No KPIs received via Flask. Using default data.")
        # Return default data if no KPIs have been received
        return {
            "kpis": [
                {
                    "downlink": {"throughput": 1920, "latency": 5.078, "jitter": 0.574, "packetLoss": 0},
                    "uplink": {"throughput": 3880, "latency": 8.586, "jitter": 1.264, "packetLoss": 0}
                },
                {
                    "downlink": {"throughput": 6800, "latency": 5.541, "jitter": 1.069, "packetLoss": 0},
                    "uplink": {"throughput": 275336, "latency": 10.301, "jitter": 1.955, "packetLoss": 0}
                }
            ]
        }

# Define the Network Slice Environment
class NetworkSliceEnv:
    def __init__(self, api_data: Dict = None):
        self.current_step = 0
        self.api_data = api_data if api_data else fetch_network_data()
        self.state = self.initial_state()
        self.min_throughput = 1000  # Example constraint (adjust as needed)
        self.max_latency = 10  # Example constraint (adjust as needed)
        self.min_allocation = 0.2  # Minimum allocation per slice
        # Dynamic throughput scale based on max throughput across slices
        self.throughput_scale = max(
            self.state['slice1']['downlink']['throughput'],
            self.state['slice1']['uplink']['throughput'],
            self.state['slice2']['downlink']['throughput'],
            self.state['slice2']['uplink']['throughput']
        )
        # Dynamic scales for penalties
        self.latency_scale = max(
            self.state['slice1']['downlink']['latency'],
            self.state['slice1']['uplink']['latency'],
            self.state['slice2']['downlink']['latency'],
            self.state['slice2']['uplink']['latency']
        )
        self.jitter_scale = max(
            self.state['slice1']['downlink']['jitter'],
            self.state['slice1']['uplink']['jitter'],
            self.state['slice2']['downlink']['jitter'],
            self.state['slice2']['uplink']['jitter']
        )

    def initial_state(self) -> Dict:
        kpis = self.api_data["kpis"]
        return {
            'slice1': {
                'downlink': kpis[0]['downlink'],
                'uplink': kpis[0]['uplink']
            },
            'slice2': {
                'downlink': kpis[1]['downlink'],
                'uplink': kpis[1]['uplink']
            },
            'primSignalStrength': -80,  # Example dynamic state (could be updated via API)
            'recCbc': 2, 'sentCbc': 1,
            'recCbdTime': 1.5, 'recThExceededDelayPkts': 5, 'sentThExceededDelayPkts': 3
        }

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.api_data = fetch_network_data()
        self.state = self.initial_state()
        return self.get_state_vector()

    def get_state_vector(self) -> np.ndarray:
        state_signals = [
            self.state['primSignalStrength'], self.state['recCbc'], self.state['sentCbc'],
            self.state['recCbdTime'], self.state['recThExceededDelayPkts'], self.state['sentThExceededDelayPkts']
        ]
        return np.array(state_signals)

    def step(self, action_dl: np.ndarray, action_ul: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.current_step += 1
        done = self.current_step >= 100  # Example episode length (adjust as needed)

        # Update KPIs dynamically based on allocations (disabled for initial training)
        # self.update_kpis(action_dl, action_ul)

        next_state = self.get_state_vector()
        reward = self.compute_reward(action_dl, action_ul)
        return next_state, reward, done, {'dl': action_dl, 'ul': action_ul}

    def update_kpis(self, action_dl: np.ndarray, action_ul: np.ndarray) -> None:
        # Example dynamic model: KPIs improve with higher allocation
        base_state = self.initial_state()
        max_kpi_value = 1e6  # Cap to prevent overflow
        for slice, alloc_dl, alloc_ul in [('slice1', action_dl[0], action_ul[0]), ('slice2', action_dl[1], action_ul[1])]:
            # Downlink KPIs
            self.state[slice]['downlink']['latency'] = min(
                base_state[slice]['downlink']['latency'] / (alloc_dl + 1e-6), max_kpi_value
            )
            self.state[slice]['downlink']['jitter'] = min(
                base_state[slice]['downlink']['jitter'] / (alloc_dl + 1e-6), max_kpi_value
            )
            self.state[slice]['downlink']['packetLoss'] = min(
                base_state[slice]['downlink']['packetLoss'] / (alloc_dl + 1e-6), max_kpi_value
            )
            self.state[slice]['downlink']['throughput'] = min(
                base_state[slice]['downlink']['throughput'] * alloc_dl, max_kpi_value
            )

            # Uplink KPIs
            self.state[slice]['uplink']['latency'] = min(
                base_state[slice]['uplink']['latency'] / (alloc_ul + 1e-6), max_kpi_value
            )
            self.state[slice]['uplink']['jitter'] = min(
                base_state[slice]['uplink']['jitter'] / (alloc_ul + 1e-6), max_kpi_value
            )
            self.state[slice]['uplink']['packetLoss'] = min(
                base_state[slice]['uplink']['packetLoss'] / (alloc_ul + 1e-6), max_kpi_value
            )
            self.state[slice]['uplink']['throughput'] = min(
                base_state[slice]['uplink']['throughput'] * alloc_ul, max_kpi_value
            )

        # Update state vector dynamically (example: signal strength improves with balanced allocations)
        self.state['primSignalStrength'] = -80 + 10 * (1 - abs(action_dl[0] - action_dl[1]))  # Example dynamic update

    def compute_reward(self, action_dl: np.ndarray, action_ul: np.ndarray) -> float:
        slice1, slice2 = self.state['slice1'], self.state['slice2']

        # Downlink reward (normalize throughput)
        dl_throughput_reward = (action_dl[0] * slice1['downlink']['throughput'] + 
                               action_dl[1] * slice2['downlink']['throughput']) / self.throughput_scale
        dl_latency_penalty = (action_dl[0] * slice1['downlink']['latency'] + 
                             action_dl[1] * slice2['downlink']['latency']) / self.latency_scale  # Use dynamic scale
        dl_jitter_penalty = (action_dl[0] * slice1['downlink']['jitter'] + 
                            action_dl[1] * slice2['downlink']['jitter']) / self.jitter_scale  # Use dynamic scale
        dl_packet_loss_penalty = (action_dl[0] * slice1['downlink']['packetLoss'] + 
                                 action_dl[1] * slice2['downlink']['packetLoss']) / 1.0  # Example scaling
        # Weight throughput more heavily to incentivize it
        dl_reward = 10.0 * dl_throughput_reward - (dl_latency_penalty + dl_jitter_penalty + dl_packet_loss_penalty)

        # Uplink reward (normalize throughput)
        ul_throughput_reward = (action_ul[0] * slice1['uplink']['throughput'] + 
                               action_ul[1] * slice2['uplink']['throughput']) / self.throughput_scale
        ul_latency_penalty = (action_ul[0] * slice1['uplink']['latency'] + 
                             action_ul[1] * slice2['uplink']['latency']) / self.latency_scale  # Use dynamic scale
        ul_jitter_penalty = (action_ul[0] * slice1['uplink']['jitter'] + 
                            action_ul[1] * slice2['uplink']['jitter']) / self.jitter_scale  # Use dynamic scale
        ul_packet_loss_penalty = (action_ul[0] * slice1['uplink']['packetLoss'] + 
                                 action_ul[1] * slice2['uplink']['packetLoss']) / 1.0  # Example scaling
        # Weight throughput more heavily to incentivize it
        ul_reward = 10.0 * ul_throughput_reward - (ul_latency_penalty + ul_jitter_penalty + ul_packet_loss_penalty)

        # Total reward (average of DL and UL for simplicity)
        reward = (dl_reward + ul_reward) / 2

        # Add soft constraints (logarithmic penalties to avoid numerical instability)
        penalty_scale = 1.0  # Reduced scale to balance with throughput reward
        throughput_violation_dl1 = max(0, self.min_throughput - action_dl[0] * slice1['downlink']['throughput'])
        throughput_violation_dl2 = max(0, self.min_throughput - action_dl[1] * slice2['downlink']['throughput'])
        throughput_violation_ul1 = max(0, self.min_throughput - action_ul[0] * slice1['uplink']['throughput'])
        throughput_violation_ul2 = max(0, self.min_throughput - action_ul[1] * slice2['uplink']['throughput'])
        latency_violation_dl1 = max(0, action_dl[0] * slice1['downlink']['latency'] - self.max_latency)
        latency_violation_dl2 = max(0, action_dl[1] * slice2['downlink']['latency'] - self.max_latency)
        latency_violation_ul1 = max(0, action_ul[0] * slice1['uplink']['latency'] - self.max_latency)
        latency_violation_ul2 = max(0, action_ul[1] * slice2['uplink']['latency'] - self.max_latency)
        allocation_violation_dl1 = max(0, self.min_allocation - action_dl[0])
        allocation_violation_dl2 = max(0, self.min_allocation - action_dl[1])
        allocation_violation_ul1 = max(0, self.min_allocation - action_ul[0])
        allocation_violation_ul2 = max(0, self.min_allocation - action_ul[1])

        reward -= penalty_scale * (
            np.log1p(throughput_violation_dl1) + np.log1p(throughput_violation_dl2) +
            np.log1p(throughput_violation_ul1) + np.log1p(throughput_violation_ul2) +
            np.log1p(latency_violation_dl1) + np.log1p(latency_violation_dl2) +
            np.log1p(latency_violation_ul1) + np.log1p(latency_violation_ul2) +
            np.log1p(allocation_violation_dl1) + np.log1p(allocation_violation_dl2) +
            np.log1p(allocation_violation_ul1) + np.log1p(allocation_violation_ul2)
        )

        return reward

# Neural Network for predicting Dirichlet distribution parameters
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 16, output_dim: int = 2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        alpha = self.softplus(self.fc2(x)) + 1e-6
        # Add a minimum value to alpha to prevent extreme distributions
        alpha = alpha + 0.1
        # Cap alpha values to prevent numerical instability
        alpha = torch.clamp(alpha, min=1e-6, max=10.0)
        return alpha

# Function to sample actions with minimum allocation constraint
def sample_constrained_action(alpha: torch.Tensor, min_allocation: float = 0.2) -> torch.Tensor:
    max_attempts = 100
    for _ in range(max_attempts):
        dirichlet_dist = Dirichlet(alpha)
        action = dirichlet_dist.sample()
        if torch.all(action >= min_allocation):
            return action
    # If no valid action is sampled, adjust the action to meet the constraint
    action = torch.clamp(action, min=min_allocation)
    # Normalize to ensure sum is 1
    action_sum = action.sum()
    if action_sum > 0:
        action = action / action_sum
    # If normalization still doesn't meet the constraint, return a default action
    if torch.any(action < min_allocation):
        action = torch.tensor([min_allocation, 1 - min_allocation])
    return action

# Training function
def train_policy(num_episodes: int = 100, seed: int = 42) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = NetworkSliceEnv()
    policy_network = PolicyNetwork()
    optimizer = optim.Adam(policy_network.parameters(), lr=0.001)  # Reduced learning rate
    gamma = 0.995
    alpha_scale = 1.0
    entropy_weight = 0.5  # Increased entropy regularization to encourage exploration

    rewards_history, loss_history, dl_allocations_slice1, dl_allocations_slice2, ul_allocations_slice1, ul_allocations_slice2 = [], [], [], [], [], []

    for episode in range(num_episodes):
        obs = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        log_probs, rewards, actions_dl, actions_ul, states = [], [], [], [], []

        while not done:
            alpha_dl = policy_network(obs) * alpha_scale
            alpha_ul = policy_network(obs) * alpha_scale

            # Check for invalid alpha values
            if torch.isnan(alpha_dl).any() or torch.isnan(alpha_ul).any():
                print(f"Warning: NaN detected in alpha values. Alpha DL: {alpha_dl}, Alpha UL: {alpha_ul}")
                break

            action_dl = sample_constrained_action(alpha_dl, env.min_allocation)
            action_ul = sample_constrained_action(alpha_ul, env.min_allocation)

            obs_, reward, done, actions = env.step(action_dl.detach().numpy(), action_ul.detach().numpy())

            if done:
                print(f"Episode: {episode} Alpha DL: {alpha_dl} Action DL: {action_dl}")
                print(f"Episode: {episode} Alpha UL: {alpha_ul} Action UL: {action_ul}")
                print(f"Got DL slice allocation; Slice1={action_dl[0].item()}, Slice2={action_dl[1].item()}")
                print(f"Got UL slice allocation; Slice1={action_ul[0].item()}, Slice2={action_ul[1].item()}")

            dirichlet_dist_dl = Dirichlet(alpha_dl)
            dirichlet_dist_ul = Dirichlet(alpha_ul)
            log_prob_dl = dirichlet_dist_dl.log_prob(action_dl)
            log_prob_ul = dirichlet_dist_ul.log_prob(action_ul)
            log_prob = log_prob_dl + log_prob_ul

            log_probs.append(log_prob)
            rewards.append(reward)
            actions_dl.append(actions['dl'])
            actions_ul.append(actions['ul'])
            states.append(obs)

            obs = torch.tensor(obs_, dtype=torch.float32)

        if torch.isnan(alpha_dl).any() or torch.isnan(alpha_ul).any():
            break  # Exit training if NaN values are detected

        # Compute discounted returns
        discounted_returns = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_returns.insert(0, cumulative_reward)
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32)

        # Normalize returns to prevent numerical instability
        if discounted_returns.std() > 0:
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-6)
        else:
            discounted_returns = discounted_returns - discounted_returns.mean()  # Center returns if std is 0

        # Accumulate gradients over the episode
        total_loss = 0
        for log_prob, G, state in zip(log_probs, discounted_returns, states):
            alpha_dl = policy_network(state) * alpha_scale
            alpha_ul = policy_network(state) * alpha_scale
            dirichlet_dist_dl = Dirichlet(alpha_dl)
            dirichlet_dist_ul = Dirichlet(alpha_ul)
            entropy_loss = -(dirichlet_dist_dl.entropy().mean() + dirichlet_dist_ul.entropy().mean()) / 2
            loss = -log_prob * G
            total_loss += loss + entropy_weight * entropy_loss

        # Perform backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)  # Increased gradient clipping
        optimizer.step()

        # Log data for visualization
        rewards_history.append(sum(rewards))
        loss_history.append(total_loss.item())
        avg_dl_allocation_slice1 = np.mean([alloc[0] for alloc in actions_dl])
        avg_dl_allocation_slice2 = np.mean([alloc[1] for alloc in actions_dl])
        avg_ul_allocation_slice1 = np.mean([alloc[0] for alloc in actions_ul])
        avg_ul_allocation_slice2 = np.mean([alloc[1] for alloc in actions_ul])
        dl_allocations_slice1.append(avg_dl_allocation_slice1)
        dl_allocations_slice2.append(avg_dl_allocation_slice2)
        ul_allocations_slice1.append(avg_ul_allocation_slice1)
        ul_allocations_slice2.append(avg_ul_allocation_slice2)

        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} - Total Reward: {sum(rewards):.2f}, Loss: {total_loss.item():.2f}")

    return rewards_history, loss_history, dl_allocations_slice1, dl_allocations_slice2, ul_allocations_slice1, ul_allocations_slice2

# Function to run multiple seeds and plot with confidence intervals
def run_experiments(num_seeds: int = 5, num_episodes: int = 100) -> None:
    all_rewards_history = []

    for seed in range(num_seeds):
        rewards_history, loss_history, dl_allocations_slice1, dl_allocations_slice2, ul_allocations_slice1, ul_allocations_slice2 = train_policy(num_episodes, seed)
        all_rewards_history.append(rewards_history)

    # Compute mean and standard deviation
    all_rewards_history = np.array(all_rewards_history)
    mean_rewards = np.mean(all_rewards_history, axis=0)
    std_rewards = np.std(all_rewards_history, axis=0)

    # Plot with confidence intervals
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.plot(mean_rewards, label='Mean Reward')
    plt.fill_between(range(num_episodes), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, label='Std Dev')
    plt.title('Total Reward over Episodes (with Confidence Intervals)')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(loss_history, 'r')
    plt.title('Loss over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')

    plt.subplot(3, 2, 3)
    plt.plot(dl_allocations_slice1, 'b', label='Slice 1')
    plt.plot(dl_allocations_slice2, 'orange', label='Slice 2')
    plt.axhline(y=0.2, color='r', linestyle='--', label='Min Allocation')
    plt.title('Average DL Allocation over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Average Allocation')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(ul_allocations_slice1, 'b', label='Slice 1')
    plt.plot(ul_allocations_slice2, 'orange', label='Slice 2')
    plt.axhline(y=0.2, color='r', linestyle='--', label='Min Allocation')
    plt.title('Average UL Allocation over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Average Allocation')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Function to run the Flask server in a separate thread
def run_flask_server():
    app.run(host='0.0.0.0', port=5004, debug=False)

# Main function to start Flask server and run experiments
if __name__ == "__main__":
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()

    # Wait a few seconds to ensure the Flask server is up
    time.sleep(10)

    # Run the experiments
    run_experiments(num_seeds=3, num_episodes=100)