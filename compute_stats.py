# compute_stats.py
import numpy as np
import json

def compute_and_save_stats(dataset_path: str, output_path: str):
    with open(dataset_path, 'r') as f:
        scenarios = json.load(f)

    all_kpis = []
    for scenario in scenarios:
        for slice_data in scenario['kpis']:
            all_kpis.append([
                slice_data['downlink']['throughput'],
                slice_data['downlink']['latency'],
                slice_data['downlink']['jitter'],
                slice_data['downlink'].get('packetLoss', 0),
                slice_data['uplink']['throughput'],
                slice_data['uplink']['latency'],
                slice_data['uplink']['jitter'],
                slice_data['uplink'].get('packetLoss', 0),
            ])

    kpi_array = np.array(all_kpis, dtype=np.float32)
    
    mean = np.mean(kpi_array, axis=0)
    std = np.std(kpi_array, axis=0)
    
    # Add a small epsilon to std to avoid division by zero
    std[std == 0] = 1e-6

    stats = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"Successfully computed and saved stats to {output_path}")
    print(f"Mean: {stats['mean']}")
    print(f"Std Dev: {stats['std']}")

if __name__ == "__main__":
    compute_and_save_stats('master_dataset_unified.json', 'kpi_stats_unified.json')
    