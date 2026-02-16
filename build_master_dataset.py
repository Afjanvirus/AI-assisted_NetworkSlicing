# build_master_dataset.py
import json
# We no longer import from kpi_scenarios
# from kpi_scenarios import KPI_DATASET 

def create_unified_dataset(synthetic_data_path: str, live_data_path: str, output_path: str):
    """
    MODIFIED: Now loads synthetic data from the new augmented JSON file.
    """
    print("--- Building Unified Master Dataset from Augmented & Live Data ---")
    
    # 1. Load the augmented synthetic data
    try:
        with open(synthetic_data_path, 'r') as f:
            synthetic_scenarios = json.load(f)
        print(f"Successfully loaded {len(synthetic_scenarios)} scenarios from '{synthetic_data_path}'.")
    except FileNotFoundError:
        print(f"Error: {synthetic_data_path} not found. Please run `generate_augmented_scenarios.py` first.")
        return

    # 2. Load the live data
    try:
        with open(live_data_path, 'r') as f:
            live_scenarios = json.load(f)
        print(f"Successfully loaded {len(live_scenarios)} scenarios from '{live_data_path}'.")
    except FileNotFoundError:
        print(f"Error: {live_data_path} not found. Please ensure it exists.")
        return

    # Add source tags
    for scenario in synthetic_scenarios:
        scenario['source'] = 'synthetic'
    for scenario in live_scenarios:
        scenario['source'] = 'live'

    # Combine them
    master_dataset = synthetic_scenarios + live_scenarios
    
    # Save the unified dataset
    with open(output_path, 'w') as f:
        json.dump(master_dataset, f, indent=2)
        
    print(f"\nSuccessfully created '{output_path}' with a total of {len(master_dataset)} scenarios.")
    print(f"({len(synthetic_scenarios)} synthetic, {len(live_scenarios)} live)")

if __name__ == "__main__":
    create_unified_dataset(
        synthetic_data_path='kpi_scenarios_augmented.json',
        live_data_path='live_data.json', 
        output_path='master_dataset_unified.json'
    )