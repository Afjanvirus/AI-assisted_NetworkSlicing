import torch 
import torch.nn as nn
from torch.distributions import Dirichlet
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(14, 8)
        self.fc2 = nn.Linear(8, 2)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        alpha = self.softplus(self.fc2(x)) + 1e-6
        alpha = torch.clamp(alpha, min=0.2)
        return alpha * 10

# Function to load only model weights
def load_model(path="trained_model.pth"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    # Initialize the model
    model = PolicyNetwork()
    
    # Load only the model weights
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {path}")
    return model

# Load the trained model during app initialization
policy_network = load_model()

@app.route('/allocate_resource', methods=['POST'])
def allocate_resource():
    """
    Use the trained model to allocate resources based on the current network state and KPIs.
    """
    try:
        data = request.json

        # Validate input structure
        if 'kpis' not in data:
            return jsonify({"error": "Missing 'kpis' in request payload"}), 400

        # Extract data
        kpis = data['kpis']
        network_state = data.get('networkState', None)  # Use None if not provided

        # Default values for missing networkState
        default_network_state = {
            "signalStrength": -80,
            "recCbc": 2,
            "sentCbC": 1,
            "recCbdTime": 1.5,
            "recThExceededDelayPkts": 5,
            "sentThExceededDelayPkts": 3
        }

        # Use networkState if provided, otherwise use default values
        network_state = network_state if network_state is not None else default_network_state

        # Construct the state vector
        state_vector = [
            network_state["signalStrength"],
            network_state["recCbc"],
            network_state["sentCbC"],
            network_state["recCbdTime"],
            network_state["recThExceededDelayPkts"],
            network_state["sentThExceededDelayPkts"],
            kpis[0]["throughput"],
            kpis[0]["latency"],
            kpis[0]["jitter"],
            kpis[0]["packetLoss"],
            kpis[1]["throughput"],
            kpis[1]["latency"],
            kpis[1]["jitter"],
            kpis[1]["packetLoss"]
        ]

        # Convert state vector to tensor
        state_tensor = torch.tensor(state_vector, dtype=torch.float32)

        # Perform inference using the trained model
        alpha = policy_network(state_tensor)
        dirichlet_dist = Dirichlet(alpha)
        action = dirichlet_dist.sample()

        return jsonify({
            "allocation": {
                "slice1": action[0].item(),
                "slice2": action[1].item()
            }
        })

    except Exception as e:
        print(f"Error occurred: {e}")  # Log the error for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5002)
