# inference_server.py
#
# A production-ready Flask server to host our network slicing AI model.
# FINAL VERSION: Includes deterministic inference logic.

import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from typing import Dict
import os
import json

# --- 1. Define Model Architecture and Helper Functions ---
# These MUST be identical to the ones used in the training script.

class PolicyNetwork(nn.Module):
    # ... (This class is unchanged) ...
    def __init__(self, input_dim: int = 16, hidden_dim: int = 256, output_dim: int = 4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        alphas = self.softplus(self.fc3(x)) + 1e-6
        return alphas # No clamp is correct based on our last training

def get_deterministic_action(alpha: torch.Tensor, min_allocation: float) -> torch.Tensor:
    """
    ### NEW DETERMINISTIC FUNCTION FOR INFERENCE ###
    Calculates the mean of the Dirichlet distribution for a stable, repeatable action.
    """
    from torch.distributions import Dirichlet
    
    # The mean of the Dirichlet is its parameters (alpha) normalized.
    action = alpha / alpha.sum()

    # Even with the mean, we must still enforce the minimum allocation constraint.
    # This logic ensures the final action is valid and deterministic.
    if torch.any(action < min_allocation):
        # If any action is too low, we adjust it and re-normalize.
        # This is a more robust way to handle the constraint deterministically.
        action = torch.clamp(action, min=min_allocation)
        action = action / action.sum()
        
    return action

# --- 2. Create a Handler Class for the Model ---

class SlicingModel:
    def __init__(self, model_path: str):
        self.min_allocation = 0.2
        self.model = PolicyNetwork(input_dim=16, hidden_dim=256, output_dim=4)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        print(f"✅ Model '{model_path}' loaded and set to evaluation mode.")

    def _get_state_vector(self, kpis: Dict) -> np.ndarray:
        # This function should use the STATS file for normalization
        # Make sure 'kpi_stats_unified.json' is in the same folder
        with open(os.path.join(os.path.dirname(__file__), 'kpi_stats_unified.json'), 'r') as f:
            kpi_stats = json.load(f)
        
        STATS_MEAN = np.array(kpi_stats['mean'], dtype=np.float32)
        STATS_STD = np.array(kpi_stats['std'], dtype=np.float32)

        s1 = kpis['kpis'][0]
        s2 = kpis['kpis'][1]
        
        raw_kpis = np.array([
            s1['downlink']['throughput'], s1['downlink']['latency'], s1['downlink']['jitter'], s1['downlink'].get('packetLoss', 0),
            s1['uplink']['throughput'], s1['uplink']['latency'], s1['uplink']['jitter'], s1['uplink'].get('packetLoss', 0),
            s2['downlink']['throughput'], s2['downlink']['latency'], s2['downlink']['jitter'], s2['downlink'].get('packetLoss', 0),
            s2['uplink']['throughput'], s2['uplink']['latency'], s2['uplink']['jitter'], s2['uplink'].get('packetLoss', 0),
        ], dtype=np.float32)
        
        mean_reshaped = np.tile(STATS_MEAN, 2)
        std_reshaped = np.tile(STATS_STD, 2)
        
        normalized_vector = (raw_kpis - mean_reshaped) / std_reshaped
        return normalized_vector

    def predict(self, kpi_data: Dict) -> Dict:
        """
        Takes raw KPI JSON data and returns a dictionary of allocations.
        """
        with torch.no_grad():
            state_vector = self._get_state_vector(kpi_data)
            obs = torch.from_numpy(state_vector)

            all_alphas = self.model(obs)
            alpha_dl, alpha_ul = torch.split(all_alphas, 2)

            # --- MODIFICATION: Use the new deterministic function ---
            action_dl = get_deterministic_action(alpha_dl, self.min_allocation)
            action_ul = get_deterministic_action(alpha_ul, self.min_allocation)

            # Format the output (using the averaging method we discussed)
            avg_slice1 = (action_dl[0].item() + action_ul[0].item()) / 2
            avg_slice2 = (action_dl[1].item() + action_ul[1].item()) / 2

            response = {
                
                "allocations": {
                    "downlink": {
                        "slice1": action_dl[0].item(),
                        "slice2": action_dl[1].item()
                    },
                    "uplink": {
                        "slice1": action_ul[0].item(),
                        "slice2": action_ul[1].item()
                    }
                }
            }
            return response

# --- 3. Set up the Flask API Server ---
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_validated_model_seed_42.pth')
model = SlicingModel(model_path=MODEL_PATH)

@app.route('/allocate_resource', methods=['POST'])
def allocate_resource():
    # ... (This part is unchanged) ...
    print("Received a new allocation request...")
    if not request.is_json:
        print("❌ Error: Request body is not JSON.")
        return jsonify({"error": "Invalid input: request body must be JSON"}), 400
    try:
        kpi_data = request.get_json()
        if "kpis" not in kpi_data or not isinstance(kpi_data["kpis"], list) or len(kpi_data["kpis"]) != 2:
            raise ValueError("Input JSON must have a 'kpis' key with a list of 2 slice objects.")
        prediction = model.predict(kpi_data)
        print(f"✅ Successfully allocated resources: {prediction}")
        return jsonify(prediction)
    except Exception as e:
        print(f"❌ Error during allocation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting the AI Slicing Inference Server...")
    app.run(host='0.0.0.0', port=5000, debug=False)