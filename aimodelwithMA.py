import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
import os
from typing import Dict, Tuple
from collections import deque  # For history buffers
import time  # For rate limiting

# --- 1. Define Model Architecture ---
# This class definition MUST be an exact copy of the ActorCritic class 
# from the final training script.

class ActorCritic(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 256, actor_output_dim: int = 4):
        super(ActorCritic, self).__init__()
        self.shared_fc1 = nn.Linear(input_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, actor_output_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.shared_fc1(x))
        x = torch.relu(self.shared_fc2(x))
        alphas = self.softplus(self.actor_head(x)) + 1e-6
        value = self.critic_head(x)
        return alphas, value

# --- 2. Create a Handler Class for the Model ---
# This class encapsulates all logic for loading, pre-processing, and running the model.
# Updated with hybrid hysteresis + thresholds + rate limiting.

class SlicingModelHandler:
    def __init__(self, model_path: str,
                 min_allocation: float = 0.2,
                 hysteresis_window: int = 3,  # Moving average window size
                 threshold_throughput_delta: float = 10000000.0,  # 10 Mbps
                 min_decision_interval: float = 4.0):  # Seconds between full decisions
        self.min_allocation = min_allocation
        self.flexible_resource = 1.0 - (self.min_allocation * 2)  # This equals 0.6
        
        # Load the model architecture
        self.model = ActorCritic(input_dim=16, hidden_dim=256, actor_output_dim=4)
        
        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        print(f"✅ Model '{model_path}' loaded successfully.")
        
        # History buffers for hysteresis (store [slice1, slice2] lists)
        self.dl_history = deque(maxlen=hysteresis_window)
        self.ul_history = deque(maxlen=hysteresis_window)
        
        # Tracking for thresholds and rate limiting
        self.last_kpis = None
        self.last_allocations = None
        self.last_decision_time = 0.0
        self.threshold_throughput_delta = threshold_throughput_delta
        self.min_decision_interval = min_decision_interval

    def _preprocess_kpis(self, kpi_data: Dict) -> np.ndarray:
        """
        Processes a raw KPI dictionary into a normalized state vector.
        This logic MUST EXACTLY match the final version from our training environment.
        """
        s1, s2 = kpi_data['kpis']
        s1_dl, s1_ul = s1['downlink'], s1['uplink']
        s2_dl, s2_ul = s2['downlink'], s2['uplink']
        
        dl_throughput_scale = max(s1_dl['throughput'], s2_dl['throughput'], 1)
        ul_throughput_scale = max(s1_ul['throughput'], s2_ul['throughput'], 1)
        latency_scale = max(s1_dl['latency'], s1_ul['latency'], s2_dl['latency'], s2_ul['latency'], 1)
        jitter_scale = max(s1_dl['jitter'], s1_ul['jitter'], s2_dl['jitter'], s2_ul['jitter'], 1)
        packet_loss_max = max(s1_dl.get('packetLoss', 0), s1_ul.get('packetLoss', 0), s2_dl.get('packetLoss', 0), s2_ul.get('packetLoss', 0))
        packet_loss_scale = packet_loss_max if packet_loss_max > 0 else 1.0
        
        state_vector = [
            s1_dl['throughput'] / dl_throughput_scale, s1_dl['latency'] / latency_scale, s1_dl['jitter'] / jitter_scale, s1_dl.get('packetLoss', 0) / packet_loss_scale,
            s1_ul['throughput'] / ul_throughput_scale, s1_ul['latency'] / latency_scale, s1_ul['jitter'] / jitter_scale, s1_ul.get('packetLoss', 0) / packet_loss_scale,
            s2_dl['throughput'] / dl_throughput_scale, s2_dl['latency'] / latency_scale, s2_dl['jitter'] / jitter_scale, s2_dl.get('packetLoss', 0) / packet_loss_scale,
            s2_ul['throughput'] / ul_throughput_scale, s2_ul['latency'] / latency_scale, s2_ul['jitter'] / jitter_scale, s2_ul.get('packetLoss', 0) / packet_loss_scale,
        ]
        
        return np.array(state_vector, dtype=np.float32)

    def _check_thresholds(self, current_kpis: Dict) -> bool:
        if self.last_kpis is None:
            return True  # Always process first request
        
        # Example: Total DL/UL throughput delta (customize, e.g., add latency/jitter if needed)
        curr_dl_total = current_kpis['kpis'][0]['downlink']['throughput'] + current_kpis['kpis'][1]['downlink']['throughput']
        last_dl_total = self.last_kpis['kpis'][0]['downlink']['throughput'] + self.last_kpis['kpis'][1]['downlink']['throughput']
        curr_ul_total = current_kpis['kpis'][0]['uplink']['throughput'] + current_kpis['kpis'][1]['uplink']['throughput']
        last_ul_total = self.last_kpis['kpis'][0]['uplink']['throughput'] + self.last_kpis['kpis'][1]['uplink']['throughput']
        
        delta_dl = abs(curr_dl_total - last_dl_total)
        delta_ul = abs(curr_ul_total - last_ul_total)
        
        print(f"DEBUG: DL Delta: {delta_dl}, UL Delta: {delta_ul} (Threshold: {self.threshold_throughput_delta})")
        return delta_dl > self.threshold_throughput_delta or delta_ul > self.threshold_throughput_delta

    def _apply_hysteresis(self, history: deque, new_alloc: list) -> list:
        history.append(new_alloc)
        if len(history) < history.maxlen:
            return new_alloc  # Use raw until buffer fills
        smoothed = np.mean(history, axis=0).tolist()
        # Normalize to sum=1.0
        total = sum(smoothed)
        smoothed = [x / total for x in smoothed] if total > 0 else new_alloc
        print(f"DEBUG: Smoothed allocation: {smoothed}")
        return smoothed

    def predict(self, kpi_data: Dict) -> Dict:
        current_time = time.time()
        
        # Check thresholds first - if significant change, prioritize processing
        thresholds_met = self._check_thresholds(kpi_data)
        
        # Rate limiting: Only skip if not time yet AND thresholds not met
        if not thresholds_met and (current_time - self.last_decision_time < self.min_decision_interval):
            print(f"DEBUG: Rate limit hit and thresholds not met; returning last allocations: {self.last_allocations}")
            return self.last_allocations or {"error": "No prior allocations available; request ignored due to rate limit"}
        
        if thresholds_met and (current_time - self.last_decision_time < self.min_decision_interval):
            print("WARNING: Rate limit hit but thresholds met (significant change detected); proceeding with computation for adaptability")
        
        with torch.no_grad():  # Disable gradients for speed and safety
            state_vector = self._preprocess_kpis(kpi_data)
            obs = torch.from_numpy(state_vector)
            
            all_alphas, _ = self.model(obs)
            alpha_dl, alpha_ul = torch.split(all_alphas, 2)

            agent_decision_dl = alpha_dl / alpha_dl.sum()
            agent_decision_ul = alpha_ul / alpha_ul.sum()

            flexible_alloc_dl = agent_decision_dl * self.flexible_resource
            flexible_alloc_ul = agent_decision_ul * self.flexible_resource
            
            raw_action_dl = (flexible_alloc_dl + self.min_allocation).tolist()
            raw_action_ul = (flexible_alloc_ul + self.min_allocation).tolist()
            
            # Apply hysteresis to raw outputs
            smoothed_dl = self._apply_hysteresis(self.dl_history, [raw_action_dl[0], raw_action_dl[1]])
            smoothed_ul = self._apply_hysteresis(self.ul_history, [raw_action_ul[0], raw_action_ul[1]])
            
            response = {
                "downlink_allocation": {
                    "slice1": round(smoothed_dl[0], 4),
                    "slice2": round(smoothed_dl[1], 4)
                },
                "uplink_allocation": {
                    "slice1": round(smoothed_ul[0], 4),
                    "slice2": round(smoothed_ul[1], 4)
                }
            }
            
            # Update tracking
            self.last_kpis = kpi_data
            self.last_allocations = response
            self.last_decision_time = current_time
            
            return response

# --- 3. Set up and Run the Flask API Server ---
app = Flask(__name__)

# IMPORTANT: Update this variable to the name of your best trained model file!
CHAMPION_MODEL_FILENAME = 'best_validated_model1.pth' 

# Load the model once when the server starts for efficiency
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, CHAMPION_MODEL_FILENAME)
    model_handler = SlicingModelHandler(model_path=MODEL_PATH)  # Pass tunables here if customizing
except FileNotFoundError:
    print("="*50)
    print(f"❌ FATAL ERROR: Model file not found at '{MODEL_PATH}'")
    print("Please make sure the trained .pth file is in the same directory as this script.")
    print("="*50)
    model_handler = None  # This will prevent the app from starting

@app.route('/allocate_resource', methods=['POST'])
def allocate_resource():
    """API endpoint to allocate resources based on network KPIs."""
    if not model_handler:
        return jsonify({"error": "Model not loaded. The server is not operational."}), 503

    print("\nReceived a new allocation request...")
    
    if not request.is_json:
        print("❌ Error: Request body is not JSON.")
        return jsonify({"error": "Invalid input: request body must be JSON"}), 400
    
    try:
        kpi_data = request.get_json()
        if "kpis" not in kpi_data or not isinstance(kpi_data["kpis"], list) or len(kpi_data["kpis"]) != 2:
            raise ValueError("Input JSON must have a 'kpis' key with a list of 2 slice objects.")

        prediction = model_handler.predict(kpi_data)
        
        print(f"✅ Successfully processed request. Nuanced Allocation: {prediction}")
        return jsonify(prediction)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

if __name__ == '__main__':
    if model_handler:
        print("🚀 Starting the Nuance-Only AI Slicing Inference Server...")
        # Use host='0.0.0.0' to make the server accessible on your local network
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("🛑 Server did not start because the model could not be loaded.")