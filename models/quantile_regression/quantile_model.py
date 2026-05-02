import torch
import torch.nn as nn
import numpy as np


# ── Quantile grid ────────────────────────────────────────────────────────────
QUANTILES = np.linspace(0.05, 0.95, 19).tolist()   # [0.05, 0.10, …, 0.95]
N_QUANTILES = len(QUANTILES)                        # 19


# ── Loss ─────────────────────────────────────────────────────────────────────
class PinballLoss(nn.Module):
    """
    Pinball / Quantile Regression Loss.

    For a single quantile τ ∈ (0, 1):
        L(τ) = τ · max(y - ŷ, 0)  +  (1 - τ) · max(ŷ - y, 0)

    Here we average over all 19 quantiles and the whole batch.

    Args:
        quantiles: list/array of τ values, default is the module-level QUANTILES.
    """
    def __init__(self, quantiles: list = None):
        super().__init__()
        q = quantiles if quantiles is not None else QUANTILES
        # shape (1, Q) so it broadcasts against (B, Q)
        self.register_buffer("quantiles", torch.tensor(q, dtype=torch.float32).unsqueeze(0))

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds:  (B, Q) - one prediction per quantile
            target: (B, 1) or (B,) - ground-truth scalar
        Returns:
            scalar mean pinball loss
        """
        target = target.view(-1, 1)          # (B, 1)  → broadcasts to (B, Q)
        errors = target - preds              # (B, Q)
        loss = torch.max(
            self.quantiles * errors,
            (self.quantiles - 1.0) * errors
        )                                    # (B, Q)
        return loss.mean()


# ── Model ────────────────────────────────────────────────────────────────────
class QuantileTempCNN(nn.Module):
    """
    1-D CNN that predicts a full quantile distribution [0.05 … 0.95] of the
    monthly maximum temperature, using only partial-month observations
    (e.g. the first 20 days) masked via a temporal mask.

    Architecture mirrors MaskTempCNN / YearMaskTempCNN but:
      • No year injection (the head takes 128 → 64 → N_QUANTILES).
      • Outputs are sorted along the quantile axis to enforce monotonicity
        (soft constraint: no extra loss term needed).

    Args:
        in_channels:  Number of spatial channels (e.g. 117 capitals).
        dropout:      Dropout rate.
        quantiles:    Quantile grid; defaults to module-level QUANTILES.
        sort_output:  If True (default) sort quantile outputs so q10 ≤ q50 ≤ q90.
    """
    def __init__(
        self,
        in_channels: int,
        dropout: float,
        quantiles: list = None,
        sort_output: bool = True,
    ):
        super().__init__()
        self.quantiles = quantiles if quantiles is not None else QUANTILES
        self.n_quantiles = len(self.quantiles)
        self.sort_output = sort_output
        self.dropout = dropout

        # ── Convolutional backbone (identical to your existing models) ────────
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=5, padding="same"),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, padding="same"),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding="same"),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding="same"),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout),
        )

        # ── Quantile head: 128 → 64 → N_QUANTILES ────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout),
            nn.Linear(64, self.n_quantiles),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, year=None) -> torch.Tensor:
        """
        Args:
            x:    (B, C, T) normalised temperature tensor
            mask: (B, C, T)  combined spatial + temporal mask  (0 = ignore)
            year: ignored, kept for API compatibility with train loop
        Returns:
            preds: (B, Q) quantile predictions, monotonically sorted if sort_output=True
        """
        # 1. Apply mask
        if mask is not None:
            x = x * mask

        # 2. Convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # 3. Mask-aware global average pooling
        if mask is not None:
            # collapse spatial dim → (B, 1, T) weight per time-step
            mask_pooled = torch.mean(mask, dim=1, keepdim=True)
            valid_count = mask_pooled.sum(dim=2, keepdim=True).clamp(min=1)
            x = (x * mask_pooled).sum(dim=2) / valid_count.squeeze(-1)
        else:
            x = torch.mean(x, dim=2)           # (B, 128)

        # 4. Quantile head
        preds = self.fc(x)                      # (B, Q)

        # 5. Enforce monotonicity: q05 ≤ q10 ≤ … ≤ q95
        if self.sort_output:
            preds, _ = torch.sort(preds, dim=-1)

        return preds                            # (B, Q)