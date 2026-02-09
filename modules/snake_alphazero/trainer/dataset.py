import torch
from torch.utils.data import Dataset
import json
import os
import glob
import logging

class SnakeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self._load_data()

    def _load_data(self):
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        logging.info(f"Found {len(files)} data files.")
        for fpath in files:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    # Expected data structure: list of dicts
                    # [{"input": [...], "policy": [...], "value": float}, ...]
                    self.samples.extend(data)
            except Exception as e:
                logging.error(f"Failed to load {fpath}: {e}")
                
        logging.info(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Parse Input: state is list of 4 channels, each is list of H*W values
        # Flatten to 1D then reshape later
        state = sample['state']  # [[c0 values], [c1 values], [c2 values], [c3 values]]
        flat_state = [val for channel in state for val in channel]
        input_data = torch.tensor(flat_state, dtype=torch.float32)
        
        policy_target = torch.tensor(sample['policy'], dtype=torch.float32)
        value_target = torch.tensor(sample['value'], dtype=torch.float32)
        
        return input_data, policy_target, value_target
