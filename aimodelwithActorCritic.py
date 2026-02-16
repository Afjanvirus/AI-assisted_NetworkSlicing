import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
import os
from typing import Dict, Tuple

# --- 1. Define Model Architecture ---
# This MUST be identical to the ActorCritic class from the training script.

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

class SlicingModelHandler:
    def __init__(self, model_path: str):
        self.min_allocation = 0.2
        
        # Load the correct model architecture
        self.model = ActorCritic(input_dim=16, hidden_dim=256, actor_output_dim=4)
        
        # Load the trained weights from our champion model
        # Use map_location to ensure it works on CPU if the model was trained on a GPU
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # **CRITICAL:** Set the model to evaluation mode.
        self.model.eval()
        print(f"✅ Model '{model_path}' loaded successfully and set to evaluation mode.")

    def _preprocess_kpis(self, kpi_data: Dict) -> np.ndarray:
        """
        Processes a raw KPI dictionary into a normalized state vector for the model.
        This logic MUST EXACTLY match the final version from our training environment.
        """
        s1 = kpi_data['kpis'][0]
        s2 = kpi_data['kpis'][1]
        s1_dl, s1_ul = s1['downlink'], s1['uplink']
        s2_dl, s2_ul = s2['downlink'], s2['uplink']
        
        # **CRITICAL**: Use the same decoupled normalization as in training
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

    def predict(self, kpi_data: Dict) -> Dict:
        """
        Takes raw KPI data, preprocesses it, and returns a deterministic allocation decision.
        """
        with torch.no_grad(): # Disable gradients for speed and safety
            state_vector = self._preprocess_kpis(kpi_data)
            obs = torch.from_numpy(state_vector)
            
            # The model returns (alphas, value). We only need alphas for inference.
            all_alphas, _ = self.model(obs)
            alpha_dl, alpha_ul = torch.split(all_alphas, 2)

            # **CRITICAL**: Use deterministic argmax for deployment, not sampling.
            # This ensures the best, most confident action is always chosen.
            dl_choice = torch.argmax(alpha_dl)
            ul_choice = torch.argmax(alpha_ul)

            action_dl = torch.tensor([self.min_allocation, self.min_allocation])
            action_ul = torch.tensor([self.min_allocation, self.min_allocation])
            
            action_dl[dl_choice] = 1.0 - self.min_allocation
            action_ul[ul_choice] = 1.0 - self.min_allocation

            response = {
                "downlink_allocation": {
                    "slice1": round(action_dl[0].item(), 4),
                    "slice2": round(action_dl[1].item(), 4)
                },
                "uplink_allocation": {
                    "slice1": round(action_ul[0].item(), 4),
                    "slice2": round(action_ul[1].item(), 4)
                },
                "model_decision": {
                    "downlink_choice": f"slice{dl_choice.item() + 1}",
                    "uplink_choice": f"slice{ul_choice.item() + 1}"
                }
            }
            return response

# --- 3. Set up the Flask API Server ---
app = Flask(__name__)

# --- ACTION: Update this to the name of your CHAMPION model file ---
CHAMPION_MODEL_FILENAME = 'slice_allocator_finalflexy_seed1.pth' # Or _seed1, _seed2, etc.

# Load the model once when the server starts
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, CHAMPION_MODEL_FILENAME)
    model_handler = SlicingModelHandler(model_path=MODEL_PATH)
except FileNotFoundError:
    print("="*50)
    print(f"❌ FATAL ERROR: Model file not found at '{MODEL_PATH}'")
    print("Please make sure you have trained a model and the .pth file is in the same directory as this script.")
    print("="*50)
    model_handler = None # Prevent the app from starting if the model is missing

@app.route('/allocate_resource', methods=['POST'])
def allocate_resource():
    """API endpoint to allocate resources."""
    if not model_handler:
        return jsonify({"error": "Model not loaded. Server is not operational."}), 503

    print("\nReceived a new allocation request...")
    
    if not request.is_json:
        print("❌ Error: Request body is not JSON.")
        return jsonify({"error": "Invalid input: request body must be JSON"}), 400
    
    try:
        kpi_data = request.get_json()
        if "kpis" not in kpi_data or not isinstance(kpi_data["kpis"], list) or len(kpi_data["kpis"]) != 2:
            raise ValueError("Input JSON must have a 'kpis' key with a list of 2 slice objects.")

        prediction = model_handler.predict(kpi_data)
        
        print(f"✅ Successfully processed request. Allocation: {prediction}")
        return jsonify(prediction)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

if __name__ == '__main__':
    if model_handler:
        print("🚀 Starting the AI Slicing Inference Server...")
        # Use `host='0.0.0.0'` to make it accessible on your local network
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("🛑 Server did not start because the model could not be loaded.")