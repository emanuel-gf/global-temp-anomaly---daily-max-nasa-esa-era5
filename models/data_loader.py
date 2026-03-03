import torch
import numpy as np
from torch.utils.data import Dataset

class BaseTempDataset(Dataset):
    """Utility base class to share normalization logic across all variants."""
    def _extract_years(self, month_years_array):
        # Extracts year as float from "YYYY-MM" strings
        years = [float(str(my)[:4]) for my in month_years_array]
        return np.array(years, dtype=np.float32)

    def _normalize(self, temp_matrices, mu, std):
        tt = torch.tensor(temp_matrices, dtype=torch.float32)
        tts = (tt - mu) / (std + 1e-8)
        return tts.numpy()

class YearTempDSMask(BaseTempDataset):
    """Dataset class with Masking and Year injection.

        Args:
            npz_path (str): Path to the .npz file containing the data. 
            The npz should contain a dictionary with keys: 'temp_matrices', 'masks', 'labels', 'month_years', 'capitals'.
        The expected data is a matrix of shape (num_samples(features), num_days)
        The feature can be for context: 100 capitals, 15 big regions etc...
        The label is the max temp for the month.

          """
    def __init__(self, npz_path, return_mask=True, exclude_2025=True, split_idx: int= -12):
        self.return_mask = return_mask
        data = np.load(npz_path, allow_pickle=True)
        
        ## which index of the array to split for training vs 2025 holdout.
        #  By default, it takes the last 12 samples as 2025 data, but can be overridden for flexibility.
        split_idx = split_idx if exclude_2025 else None

        self.temp_matrices_raw = data['temp_matrices'][:split_idx]
        self.masks = data['masks'][:split_idx]
        self.labels = data['labels'][:split_idx]
        self.month_years = data['month_years'][:split_idx]
        self.capitals = data['capitals'].astype(str)
        self.years = self._extract_years(self.month_years)

        # Holdout 2025 logic
        self.temp_matrices_2025_raw = data['temp_matrices'][split_idx:]
        self.masks_2025 = data['masks'][split_idx:]
        self.labels_2025 = data['labels'][split_idx:]
        self.month_years_2025 = data['month_years'][split_idx:]
        self.years_2025 = self._extract_years(self.month_years_2025)

        # Normalize using only training stats to prevent leakage
        self.mu = torch.tensor(self.temp_matrices_raw, dtype=torch.float32).mean(dim=0)
        self.std = torch.tensor(self.temp_matrices_raw, dtype=torch.float32).std(dim=0)
        
        self.temp_matrices = self._normalize(self.temp_matrices_raw, self.mu, self.std)
        self.temp_matrices_2025 = self._normalize(self.temp_matrices_2025_raw, self.mu, self.std)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.temp_matrices[idx])
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        year = torch.tensor([self.years[idx]], dtype=torch.float32)
        if self.return_mask:
            mask = torch.FloatTensor(self.masks[idx])
            return x, y, mask, year
        return x, y, year

    def get_2025_dataset(self):
        return TempDS2025(self.temp_matrices_2025, self.masks_2025, self.labels_2025, 
                          self.month_years_2025, self.years_2025, self.return_mask)

class TempDS2025(Dataset):
    """Holdout dataset for 2025 testing."""
    def __init__(self, temp_matrices, masks, labels, month_years, years, return_mask=True):
        self.temp_matrices = temp_matrices
        self.masks = masks
        self.labels = labels
        self.month_years = month_years
        self.years = years
        self.return_mask = return_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.temp_matrices[idx])
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        year = torch.tensor([self.years[idx]], dtype=torch.float32)
        if self.return_mask:
            mask = torch.FloatTensor(self.masks[idx])
            return x, y, mask, year
        return x, y, year

class YearTempDSMask_Finetun(BaseTempDataset):
    """Dataset for fine-tuning using pre-existing mu/std."""
    def __init__(self, npz_path, mu, std, return_mask=True):
        self.return_mask = return_mask
        self.mu, self.std = mu, std
        data = np.load(npz_path, allow_pickle=True)
        
        self.temp_matrices_raw = data['temp_matrices']
        self.masks = data['masks']
        self.labels = data['labels']
        self.month_years = data['month_years']
        self.years = self._extract_years(self.month_years)
        self.temp_matrices = self._normalize(self.temp_matrices_raw, self.mu, self.std)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.temp_matrices[idx])
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        year = torch.tensor([self.years[idx]], dtype=torch.float32)
        if self.return_mask:
            mask = torch.FloatTensor(self.masks[idx])
            return x, y, mask, year
        return x, y, year