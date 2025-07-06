# src/dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset
from src.preprocessing import extract_mfcc

class EmotionDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Extract MFCC features from the audio file
        mfcc = extract_mfcc(self.file_paths[idx])

        # Handle failed extraction (None returned)
        if mfcc is None:
            mfcc = np.zeros((40, 174))  # Fallback zero-tensor

        label = self.labels[idx]

        # Convert to torch tensor
        x = torch.tensor(mfcc, dtype=torch.float32).transpose(0, 1)  # shape: (time_steps, features)
        y = torch.tensor(label, dtype=torch.long)

        return x, y
