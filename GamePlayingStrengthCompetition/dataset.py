import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataPath, transform=None):
        self.dataPath = dataPath
        self.transform = transform
        self.data = pd.read_csv(dataPath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert X and y to tensor and ensure they have the same dtype (e.g., float32)
        X = torch.tensor(self.data.iloc[idx, 1:3].values, dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float32)
        y = y.unsqueeze(0)
        # If using a transform, apply it to X
        if self.transform:
            X = self.transform(X)

        return X, y
