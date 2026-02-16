# generate_augmented_scenarios.py
import json
import random
from kpi_scenarios import KPI_DATASET # Imports your 5 base scenarios

def augment_scenario(base_scenario, scenario_id):
    """
    Takes a single scenario and returns a new, slightly modified version.
    """
    new_scenario = {
        "scenario_name": f"{base_scenario['scenario_name']} (Augmented #{scenario_id})",
        "kpis": [
            {"downlink": {}, "uplink": {}},
            {"downlink": {}, "uplink": {}}
        ]
    }
    
    for i in range(2): # For each slice
        for direction in ['downlink', 'uplink']: # For DL and UL
            for kpi, value in base_scenario['kpis'][i][direction].items():
                if kpi == 'throughput':
                    # Add/subtract up to 20% of the original value
                    noise = random.uniform(-0.20, 0.20)
                elif kpi == 'latency' or kpi == 'jitter':
                    # Add/subtract up to 15%
                    noise = random.uniform(-0.15, 0.15)
                elif kpi == 'packetLoss':
                    # Add/subtract a small absolute value (e.g., up to 0.05)
                    noise_abs = random.uniform(-0.05, 0.05)
                    # We apply noise as absolute, not relative, for packet loss
                    new_value = max(0, value + noise_abs) # Ensure it doesn't go below 0
                    new_scenario['kpis'][i][direction][kpi] = new_value
                    continue # Skip the relative noise calculation below
                
                # Apply the relative noise
                new_value = value * (1 + noise)
                # Ensure values don't go below zero
                new_scenario['kpis'][i][direction][kpi] = max(0, new_value)
                
    return new_scenario


if __name__ == "__main__":
    NUM_SCENARIOS_TO_GENERATE = 2000 # Let's create a large dataset
    
    augmented_dataset = []
    for i in range(NUM_SCENARIOS_TO_GENERATE):
        # Pick one of the 5 base scenarios at random to be the template
        base_scenario_template = random.choice(KPI_DATASET)
        new_aug_scenario = augment_scenario(base_scenario_template, i + 1)
        augmented_dataset.append(new_aug_scenario)
        
    output_path = 'kpi_scenarios_augmented.json'
    with open(output_path, 'w') as f:
        json.dump(augmented_dataset, f, indent=2)
        
    print(f"Successfully generated {len(augmented_dataset)} augmented scenarios.")
    print(f"Saved to: {output_path}")
