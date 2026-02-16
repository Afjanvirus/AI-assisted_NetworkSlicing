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
import logging

# --- Flask Server Setup ---
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)
latest_kpis = None

@app.route('/kpis', methods=['POST'])
def receive_kpis():
    global latest_kpis
    data = request.get_json()
    if not data or 'kpis' not in data:
        return jsonify({"error": "Invalid JSON format. Expected 'kpis' key."}), 400
    latest_kpis = data
    print("\n--- New KPIs received from Flask ---\n")
    return jsonify({"message": "KPIs received successfully"}), 200

def run_flask_server():
    app.run(host='0.0.0.0', port=5004, debug=False)
# --- End of Flask Setup ---


def fetch_network_data() -> Dict:
    global latest_kpis
    if latest_kpis is not None:
        return latest_kpis
    else:
        print("No KPIs received via Flask. Using default data for now.")
        return {
            "kpis": [
                {"downlink": {"throughput": 1920, "latency": 5.078, "jitter": 0.574, "packetLoss": 0},
                 "uplink": {"throughput": 3880, "latency": 8.586, "jitter": 1.264, "packetLoss": 0}},
                {"downlink": {"throughput": 6800, "latency": 5.541, "jitter": 1.069, "packetLoss": 0},
                 "uplink": {"throughput": 275336, "latency": 10.301, "jitter": 1.955, "packetLoss": 0}}
            ]
        }

# Define the Network Slice Environment
class NetworkSliceEnv:
    def __init__(self, api_data: Dict = None):
        self.current_step = 0
        self.api_data = api_data if api_data else fetch_network_data()
        self.state = self.initial_state()
        self.min_allocation = 0.2
        self.update_scales()

    def update_scales(self):
        s1_dl, s1_ul = self.state['slice1']['downlink'], self.state['slice1']['uplink']
        s2_dl, s2_ul = self.state['slice2']['downlink'], self.state['slice2']['uplink']
        self.throughput_scale = max(s1_dl['throughput'], s1_ul['throughput'], s2_dl['throughput'], s2_ul['throughput'], 1)
        self.latency_scale = max(s1_dl['latency'], s1_ul['latency'], s2_dl['latency'], s2_ul['latency'], 1)
        self.jitter_scale = max(s1_dl['jitter'], s1_ul['jitter'], s2_dl['jitter'], s2_ul['jitter'], 1)
        self.packet_loss_scale = max(s1_dl.get('packetLoss', 0), s1_ul.get('packetLoss', 0), s2_dl.get('packetLoss', 0), s2_ul.get('packetLoss', 0))
        if self.packet_loss_scale == 0: self.packet_loss_scale = 1.0

    def initial_state(self) -> Dict:
        kpis = self.api_data["kpis"]
        return {
            'slice1': {'downlink': kpis[0]['downlink'], 'uplink': kpis[0]['uplink']},
            'slice2': {'downlink': kpis[1]['downlink'], 'uplink': kpis[1]['uplink']},
        }

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.api_data = fetch_network_data()
        self.state = self.initial_state()
        self.update_scales()
        return self.get_state_vector()

    def get_state_vector(self) -> np.ndarray:
        s1_dl, s1_ul = self.state['slice1']['downlink'], self.state['slice1']['uplink']
        s2_dl, s2_ul = self.state['slice2']['downlink'], self.state['slice2']['uplink']
        state_vector = [
            s1_dl['throughput'] / self.throughput_scale, s1_dl['latency'] / self.latency_scale, s1_dl['jitter'] / self.jitter_scale, s1_dl.get('packetLoss', 0) / self.packet_loss_scale,
            s1_ul['throughput'] / self.throughput_scale, s1_ul['latency'] / self.latency_scale, s1_ul['jitter'] / self.jitter_scale, s1_ul.get('packetLoss', 0) / self.packet_loss_scale,
            s2_dl['throughput'] / self.throughput_scale, s2_dl['latency'] / self.latency_scale, s2_dl['jitter'] / self.jitter_scale, s2_dl.get('packetLoss', 0) / self.packet_loss_scale,
            s2_ul['throughput'] / self.throughput_scale, s2_ul['latency'] / self.latency_scale, s2_ul['jitter'] / self.jitter_scale, s2_ul.get('packetLoss', 0) / self.packet_loss_scale,
        ]
        return np.array(state_vector, dtype=np.float32)

    def step(self, action_dl: np.ndarray, action_ul: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self.current_step += 1
        done = self.current_step >= 100
        next_state = self.get_state_vector()
        reward = self.compute_reward(action_dl, action_ul)
        return next_state, reward, done, {'dl': action_dl, 'ul': action_ul}

    def compute_reward(self, action_dl: np.ndarray, action_ul: np.ndarray) -> float:
        slice1, slice2 = self.state['slice1'], self.state['slice2']
        throughput_weight, penalty_weight = 10.0, 1.0
        dl_thr = (action_dl[0] * slice1['downlink']['throughput'] + action_dl[1] * slice2['downlink']['throughput']) / self.throughput_scale
        dl_lat = (action_dl[0] * slice1['downlink']['latency'] + action_dl[1] * slice2['downlink']['latency']) / self.latency_scale
        dl_jit = (action_dl[0] * slice1['downlink']['jitter'] + action_dl[1] * slice2['downlink']['jitter']) / self.jitter_scale
        dl_pl = (action_dl[0] * slice1['downlink'].get('packetLoss', 0) + action_dl[1] * slice2['downlink'].get('packetLoss', 0)) / self.packet_loss_scale
        dl_reward = throughput_weight * dl_thr - penalty_weight * (dl_lat + dl_jit + dl_pl)
        ul_thr = (action_ul[0] * slice1['uplink']['throughput'] + action_ul[1] * slice2['uplink']['throughput']) / self.throughput_scale
        ul_lat = (action_ul[0] * slice1['uplink']['latency'] + action_ul[1] * slice2['uplink']['latency']) / self.latency_scale
        ul_jit = (action_ul[0] * slice1['uplink']['jitter'] + action_ul[1] * slice2['uplink']['jitter']) / self.jitter_scale
        ul_pl = (action_ul[0] * slice1['uplink'].get('packetLoss', 0) + action_ul[1] * slice2['uplink'].get('packetLoss', 0)) / self.packet_loss_scale
        ul_reward = throughput_weight * ul_thr - penalty_weight * (ul_lat + ul_jit + ul_pl)
        return (dl_reward + ul_reward) / 2

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, output_dim: int = 4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        alphas = self.softplus(self.fc3(x)) + 1e-6
        return torch.clamp(alphas, min=0.1)

def sample_constrained_action(alpha: torch.Tensor, min_allocation: float) -> torch.Tensor:
    try:
        action = Dirichlet(alpha).sample()
        if torch.all(action >= min_allocation):
            return action
        else:
            if action[0] < min_allocation:
                delta = min_allocation - action[0]
                action[0] = min_allocation
                action[1] -= delta
            elif action[1] < min_allocation:
                delta = min_allocation - action[1]
                action[1] = min_allocation
                action[0] -= delta
            action = torch.clamp(action, min=min_allocation)
            return action / action.sum()
    except ValueError:
        return torch.tensor([0.5, 0.5])

# Training function
def train_policy(num_episodes: int, seed: int) -> Tuple[List[float], ...]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = NetworkSliceEnv()
    policy_network = PolicyNetwork(input_dim=len(env.get_state_vector()))
    optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
    gamma = 0.99
    entropy_weight = 0.01

    results = ([], [], [], [], [], [])

    for episode in range(num_episodes):
        obs = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        log_probs_list, rewards, actions_dl, actions_ul = [], [], [], []

        while not done:
            if torch.isnan(obs).any(): break
            
            all_alphas = policy_network(obs)
            alpha_dl, alpha_ul = torch.split(all_alphas, 2)

            action_dl = sample_constrained_action(alpha_dl, env.min_allocation)
            action_ul = sample_constrained_action(alpha_ul, env.min_allocation)

            obs_, reward, done, actions = env.step(action_dl.detach().numpy(), action_ul.detach().numpy())
            
            # --- MODIFICATION: Add detailed end-of-episode printing ---
            if done:
                print(f"Episode: {episode} Alpha DL: {alpha_dl} Action DL: {action_dl}")
                print(f"Episode: {episode} Alpha UL: {alpha_ul} Action UL: {action_ul}")
            # --- END OF MODIFICATION ---
            
            log_prob_dl = Dirichlet(alpha_dl).log_prob(action_dl)
            log_prob_ul = Dirichlet(alpha_ul).log_prob(action_ul)
            log_prob = log_prob_dl + log_prob_ul

            log_probs_list.append(log_prob)
            rewards.append(reward)
            actions_dl.append(actions['dl'])
            actions_ul.append(actions['ul'])
            obs = torch.tensor(obs_, dtype=torch.float32)
        
        if torch.isnan(obs).any(): continue

        discounted_returns = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + gamma * cumulative_reward
            discounted_returns.insert(0, cumulative_reward)
        
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32)
        if len(discounted_returns) > 1:
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-9)

        total_loss = 0
        policy_losses = []
        for log_prob, G in zip(log_probs_list, discounted_returns):
            policy_losses.append(-log_prob * G)
        
        total_loss = torch.stack(policy_losses).mean()

        if total_loss != 0:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
            optimizer.step()

        # Log data
        results[0].append(sum(rewards))
        results[1].append(total_loss.item() if total_loss != 0 else 0)
        results[2].append(np.mean([a[0] for a in actions_dl]))
        results[3].append(np.mean([a[1] for a in actions_dl]))
        results[4].append(np.mean([a[0] for a in actions_ul]))
        results[5].append(np.mean([a[1] for a in actions_ul]))

        if episode % 10 == 0:
            print(f"Episode {episode:3d} | Total Reward: {sum(rewards):8.2f} | Loss: {results[1][-1]:8.2f}")

    return results

def run_experiments(num_seeds: int = 2, num_episodes: int = 100):
    all_rewards_history = []
    final_results = None

    for seed in range(num_seeds):
        print(f"\n--- Running Seed {seed + 1}/{num_seeds} ---")
        results = train_policy(num_episodes, seed)
        if results and results[0]:
            all_rewards_history.append(results[0])
            final_results = results

    if not final_results:
        print("Training failed for all seeds.")
        return

    all_rewards_history = np.array(all_rewards_history)
    mean_rewards = np.mean(all_rewards_history, axis=0)
    std_rewards = np.std(all_rewards_history, axis=0)

    _, loss_history, dl_s1, dl_s2, ul_s1, ul_s2 = final_results
    
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 2, 1)
    plt.plot(mean_rewards, label='Mean Reward')
    plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3, label='Std Dev')
    plt.title('Total Reward over Episodes'); plt.xlabel('Episodes'); plt.ylabel('Total Reward'); plt.legend(); plt.grid(True)
    plt.subplot(3, 2, 2)
    plt.plot(loss_history, 'r', label='Loss')
    plt.title('Loss over Episodes'); plt.xlabel('Episodes'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(3, 2, 3)
    plt.plot(dl_s1, 'b', label='Slice 1 DL'); plt.plot(dl_s2, 'orange', label='Slice 2 DL')
    plt.axhline(y=0.2, color='r', linestyle='--', label='Min Allocation')
    plt.ylim(0, 1); plt.title('Average DL Allocation'); plt.xlabel('Episodes'); plt.ylabel('Allocation'); plt.legend(); plt.grid(True)
    plt.subplot(3, 2, 4)
    plt.plot(ul_s1, 'g', label='Slice 1 UL'); plt.plot(ul_s2, 'purple', label='Slice 2 UL')
    plt.axhline(y=0.2, color='r', linestyle='--', label='Min Allocation')
    plt.ylim(0, 1); plt.title('Average UL Allocation'); plt.xlabel('Episodes'); plt.ylabel('Allocation'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    print("Flask server started. Send KPIs to http://localhost:5004/kpis")
    time.sleep(10)  # Allow time for Flask server to start
    print("Starting training...")
    run_experiments(num_seeds=2, num_episodes=200)