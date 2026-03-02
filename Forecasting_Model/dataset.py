import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import random

class BalancedFlashFloodDataset(Dataset):
    def __init__(self, csv_path, seq_length=24):
        print("Loading and Balancing Dataset...")
        df = pd.read_csv(csv_path)
        
        # 1. Scale Features
        self.scaler = StandardScaler()
        feature_cols = ['Precipitation_mm', 'Soil_Moisture', 'Temperature_C', 'Elevation_m']
        scaled_features = self.scaler.fit_transform(df[feature_cols])
        targets = df['Flash_Flood_Risk'].values
        
        self.valid_sequences = []
        self.labels = []
        
        flood_indices = []
        no_flood_indices = []
        
        # 2. Extract 24-hour sequences
        for i in range(len(df) - seq_length):
            if targets[i + seq_length] == 1:
                flood_indices.append(i)
            else:
                no_flood_indices.append(i)
                
        # --- TRACKING ORIGINAL STATS ---
        num_floods = len(flood_indices)
        num_safe = len(no_flood_indices)
        original_total = num_floods + num_safe
        
        print(f"\n--- RAW DATA STATS ---")
        print(f"Total 24h Sequences: {original_total:,}")
        print(f"Original Flood Events: {num_floods:,}")
        print(f"Original Safe Events:  {num_safe:,}")
                
        # 3. Balance 50/50
        if num_floods == 0:
            raise ValueError("No flood events found in dataset! Check thresholds.")
            
        # Randomly sample the safe indices to match the flood indices
        sampled_no_flood_indices = random.sample(no_flood_indices, num_floods)
        all_indices = flood_indices + sampled_no_flood_indices
        random.shuffle(all_indices)
        
        for idx in all_indices:
            self.valid_sequences.append(scaled_features[idx : idx + seq_length])
            self.labels.append(targets[idx + seq_length])
            
        # --- TRACKING DROPPED STATS ---
        dropped_sequences = num_safe - num_floods
        
        print(f"\n--- BALANCING STATS ---")
        print(f"DROPPED Safe Sequences: {dropped_sequences:,} (Ignored to achieve 50/50 balance)")
        print(f"Final Dataset Size:     {len(self.labels):,}")
        print(f"Final Split -> Floods: {int(sum(self.labels)):,} | Safe: {int(len(self.labels) - sum(self.labels)):,}\n")

    def __len__(self):
        return len(self.valid_sequences)

    def __getitem__(self, idx):
        x_seq = self.valid_sequences[idx]
        y_label = self.labels[idx]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.float32)
    
    # --- ADD THIS TO THE BOTTOM OF YOUR FILE ---

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Make sure this matches the file you generated in Step 1
    CSV_FILENAME = "global_flash_flood_data_decade.csv"
    
    try:
        # 1. Initialize the dataset
        dataset = BalancedFlashFloodDataset(csv_path=CSV_FILENAME, seq_length=24)
        
        # 2. Wrap it in a DataLoader
        # batch_size=64 means the model will look at 64 different 24-hour sequences at once
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # 3. Fetch a single batch to test the pipeline
        X_batch, y_batch = next(iter(dataloader))
        
        print(f"\n--- DATALOADER TEST ---")
        # X shape should be: [64, 24, 4] -> [Batch Size, Sequence Length, Num Features]
        print(f"Batch X shape: {X_batch.shape} -> [Batch Size, Sequence Length, Features]")
        # y shape should be: [64] -> [Batch Size]
        print(f"Batch y shape: {y_batch.shape} -> [Batch Size]")
        
        print(f"\nSample labels from this batch (Should be a mix of 0.0 and 1.0):")
        print(y_batch[:15].tolist())
        print("Success! The tensors are ready for the LSTM.")
        
    except FileNotFoundError:
        print(f"\n[ERROR] Could not find '{CSV_FILENAME}'.")
        print("Make sure you ran the data generation script first and the file is in the same folder.")