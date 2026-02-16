# create_data_splits.py
import json
import random

def perform_split(master_dataset_path: str):
    """
    Splits the master dataset into train, validation, and test sets.
    - Test Set = 100% of live data.
    - Train/Val Sets = A split of the synthetic data.
    """
    print("--- Creating Final Data Splits ---")
    
    with open(master_dataset_path, 'r') as f:
        master_dataset = json.load(f)

    # 1. Separate data by source
    synthetic_data = [s for s in master_dataset if s['source'] == 'synthetic']
    live_data = [s for s in master_dataset if s['source'] == 'live']
    
    print(f"Found {len(synthetic_data)} synthetic scenarios and {len(live_data)} live scenarios.")

    # 2. The Test Set is all the live data
    test_set = live_data
    
    # 3. Split the synthetic data into training and validation
    random.shuffle(synthetic_data)
    val_size = int(0.15 * len(synthetic_data)) # 15% of synthetic for validation
    
    validation_set = synthetic_data[:val_size]
    training_set = synthetic_data[val_size:]
    
    # 4. Save all three files
    with open('train_set.json', 'w') as f:
        json.dump(training_set, f)
    with open('validation_set.json', 'w') as f:
        json.dump(validation_set, f)
    with open('test_set.json', 'w') as f:
        json.dump(test_set, f)

    print("\nSuccessfully created data splits:")
    print(f"  - train_set.json:      {len(training_set)} scenarios (synthetic)")
    print(f"  - validation_set.json: {len(validation_set)} scenarios (synthetic)")
    print(f"  - test_set.json:       {len(test_set)} scenarios (live data)")

if __name__ == "__main__":
    perform_split('master_dataset_unified.json')