from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MaskTempCNN(nn.Module):
    """1D-CNN with Masking support (Standard version from your notebook)."""
    def __init__(self, in_channels, dropout):
        super().__init__()
        self.dropout = dropout

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, padding='same'),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x,year=None, mask=None):
        if mask is not None:
            x = x * mask

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if mask is not None:
            mask_pooled = torch.mean(mask, dim=1, keepdim=True)
            valid_count = mask_pooled.sum(dim=2, keepdim=True).clamp(min=1)
            x = (x * mask_pooled).sum(dim=2) / valid_count.squeeze(-1)
        else:
            x = torch.mean(x, dim=2)

        return self.fc(x)

class YearMaskTempCNN(nn.Module):
    def __init__(self, in_channels,
                 dropout, 
                 normalize_max_year = 2028,
                normalize_min_year = 1980):
        """
        Args:
            in_channels: Number of input channels e.g 117 capitals
            dropout: Dropout rate for regularization
            normalize_max_year: Max year for normalization (e.g., 2028 to give some headroom beyond 2025)
            normalize_min_year: Min year for normalization (e.g.,
        """
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.normalize_max_year = normalize_max_year
        self.normalize_min_year = normalize_min_year

        # --- Convolutional Backbone ---
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=self.dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, padding='same'),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=self.dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=self.dropout)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=self.dropout)
        )

        # --- Fully Connected Head ---
        # Note: input is now 128 (CNN features) + 1 (Year feature)
        self.fc1 = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=self.dropout)
        )
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, year, mask=None):
        """
        x: (batch, 117, 31)
        year: (batch, 1) - Raw year values (e.g., 2025)
        mask: (batch, 117, 31)
        """
        # 1. Apply mask
        if mask is not None:
            x = x * mask

        # 2. Forward through Convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # 3. Global Pooling (Mask-aware)
        if mask is not None:
            mask_pooled = torch.mean(mask, dim=1, keepdim=True)
            valid_count = mask_pooled.sum(dim=2, keepdim=True).clamp(min=1)
            x = (x * mask_pooled).sum(dim=2) / valid_count.squeeze(-1)
        else:
            x = torch.mean(x, dim=2)

        # 4. Normalize Year and Inject
        # Normalizing to [0, 1] range based on your data span (1980-2026)
        # This prevents the raw '2025' value from overwhelming the weights
        year_norm = (year - self.normalize_min_year) / (self.normalize_max_year - self.normalize_min_year)

        # Concatenate: (batch, 128) + (batch, 1) -> (batch, 129)
        x = torch.cat((x, year_norm), dim=1)

        # 5. Final Regression
        x = self.fc1(x)
        x = self.fc2(x)

        return x
    

class LinearTrend:
    """
    Fits, removes and restores a linear temperature trend.
    Designed to be baked into Detrend_MaskTempCNN as frozen buffers.

    Fit on the FULL dataset since the goal is 2026 extrapolation
    """

    def __init__(self):
        self.slope: float | None     = None
        self.intercept: float | None = None
        self._fitted: bool           = False

    def fit(self, years: np.ndarray, labels: np.ndarray) -> "LinearTrend":
        reg = LinearRegression()
        reg.fit(years.reshape(-1, 1), labels.reshape(-1, 1))
        self.slope     = float(reg.coef_[0][0])
        self.intercept = float(reg.intercept_[0])
        self._fitted   = True
        print(f"[LinearTrend] slope={self.slope:.5f} °C/yr  "
                    f"intercept={self.intercept:.4f} °C  "
                    f"→ 2026 baseline={self.slope * 2026 + self.intercept:.4f} °C")
        return self

    def remove(self, labels: np.ndarray, years: np.ndarray) -> np.ndarray:
        self._check()
        return (labels - (self.slope * years + self.intercept)).astype(np.float32)

    def restore(self,
                predictions: np.ndarray | torch.Tensor,
                years: np.ndarray) -> np.ndarray:
        self._check()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        return predictions.squeeze() + (self.slope * years + self.intercept)

    def state_dict(self) -> dict:
        self._check()
        return {"slope": self.slope, "intercept": self.intercept}

    def load_state_dict(self, state: dict) -> "LinearTrend":
        self.slope     = float(state["slope"])
        self.intercept = float(state["intercept"])
        self._fitted   = True
        return self

    def _check(self):
        if not self._fitted:
            raise RuntimeError("LinearTrend not fitted. Call .fit(years, labels) first.")

    def __repr__(self):
        if not self._fitted:
            return "LinearTrend(unfitted)"
        return (f"LinearTrend(slope={self.slope:.5f}, "
                f"intercept={self.intercept:.4f}, "
                f"2026_baseline={self.slope * 2026 + self.intercept:.4f} °C)")


class Detrend_MaskTempCNN(nn.Module):
    """
    MaskTempCNN that predicts temperature RESIDUALS after removing a
    linear trend fit on the full ERA5 dataset (1981-2025).

    The trend (slope + intercept) are registered as frozen buffers so they
    travel automatically with state_dict()

    Forward signature: (x, year, mask=mask)  ← identical to YearMaskTempCNN
    Forward output:    residual prediction    ← call restore_trend() after inference
    """

    def __init__(self, in_channels: int, dropout: float,
                 slope: float = 0.0, intercept: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.dropout     = dropout

        # Frozen — saved/loaded with state_dict() automatically
        self.register_buffer("slope",     torch.tensor(slope,     dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(intercept, dtype=torch.float32))

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.01), nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, padding='same'),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.01), nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.01), nn.Dropout(dropout)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.01), nn.Dropout(dropout)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128, 64), nn.LeakyReLU(0.01), nn.Dropout(dropout)
        )
        self.fc2 = nn.Linear(64, 1)

    @classmethod
    def from_dataset(cls, dataset, 
                     in_channels: int,
                     dropout: float) -> "Detrend_MaskTempCNN":
        """
        Fit trend on the FULL dataset then build the model.
        Call this once before the Monte Carlo loop.

        Usage:
            base_model = Detrend_MaskTempCNN.from_dataset(
                dataset_era5,
                in_channels=config['in_channels'],
                dropout=config['dropout']
            ).to(DEVICE)
        """
        trend = LinearTrend().fit(dataset.years, dataset.labels)
        print(f"[Detrend_MaskTempCNN] {trend}")
        return cls(in_channels=in_channels, dropout=dropout,
                   slope=trend.slope, intercept=trend.intercept)

    # ------------------------------------------------------------------

    def remove_trend(self, y: torch.Tensor,
                     year: torch.Tensor) -> torch.Tensor:
        """y (real °C labels) → residuals. Used inside DetrendLoaderWrapper."""
        return y - (self.slope * year + self.intercept)

    def restore_trend(self, pred: torch.Tensor,
                      year: torch.Tensor) -> torch.Tensor:
        """Residual predictions → real °C. Call after inference."""
        return pred + (self.slope * year + self.intercept)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                year: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if mask is not None:
            mask_pooled = torch.mean(mask, dim=1, keepdim=True)
            valid_count = mask_pooled.sum(dim=2, keepdim=True).clamp(min=1)
            x = (x * mask_pooled).sum(dim=2) / valid_count.squeeze(-1)
        else:
            x = torch.mean(x, dim=2)

        x = self.fc1(x)
        x = self.fc2(x)
        return x


   
class DetrendLoaderWrapper:
    """
    Wraps any DataLoader and detrends y on every batch using the model's
    frozen buffers. Completely transparent to train_model and evaluate_model
    — no changes needed to either function.

    The wrapper is re-created each iteration from the same base loaders,
    so the trend params are always consistent with the loaded checkpoint.
    """

    def __init__(self, loader: DataLoader, 
                 model: Detrend_MaskTempCNN):
        self.loader = loader
        self.model  = model

    def __iter__(self):
        slope     = self.model.slope.cpu()
        intercept = self.model.intercept.cpu()
        for x, y, mask, year in self.loader:
            y_detrended = y - (slope * year + intercept)
            yield x, y_detrended, mask, year

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)