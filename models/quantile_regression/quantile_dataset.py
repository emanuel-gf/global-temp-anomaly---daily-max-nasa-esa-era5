import torch
import numpy as np
from torch.utils.data import Dataset


class BaseTempDataset(Dataset):
    """Utility base class to share normalization logic across all variants."""
    def _extract_years(self, month_years_array):
        years = [float(str(my)[:4]) for my in month_years_array]
        return np.array(years, dtype=np.float32)

    def _normalize(self, temp_matrices, mu, std):
        tt = torch.tensor(temp_matrices, dtype=torch.float32)
        tts = (tt - mu) / (std + 1e-8)
        return tts.numpy()


class TemporalMaskDataset(BaseTempDataset):
    """
    Dataset for partial-month quantile forecasting.

    Combines two independent masks:
      - Spatial mask  : already present in the .npz (missing stations / NaNs).
      - Temporal mask : zeros out every day AFTER `observed_days`, simulating
                        the "only first N days are known" scenario at entry time.

    Combined mask: combined = spatial_mask AND temporal_mask
    A time-step is valid only if the station had data AND the day index is observed.

    Args:
        npz_path (str):         Path to the .npz archive.
        observed_days (int):    How many leading days to keep visible (default 20).
        return_mask (bool):     Always return the combined mask (default True).
        exclude_end_set (bool): Hold out the last `|split_idx|` samples (default True).
        split_idx (int):        Negative index boundary for train / holdout split
                                (default -12, i.e. last 12 months).
    """

    def __init__(
        self,
        npz_path: str,
        observed_days: int = 20,
        return_mask: bool = True,
        exclude_end_set: bool = True,
        split_idx: int = -12,
    ):
        # ── Input assertions ──────────────────────────────────────────────────
        assert isinstance(npz_path, str) and npz_path.endswith(".npz"), (
            f"npz_path must be a path string ending in '.npz', got: {npz_path!r}"
        )
        assert isinstance(observed_days, int) and observed_days > 0, (
            f"observed_days must be a positive int, got: {observed_days!r}"
        )
        assert isinstance(split_idx, int) and split_idx < 0, (
            f"split_idx must be a negative int (e.g. -12), got: {split_idx!r}"
        )

        self.observed_days = observed_days
        self.return_mask   = return_mask

        data = np.load(npz_path, allow_pickle=True)
        all_month_years = data["month_years"]
        n_total = len(all_month_years)

        assert abs(split_idx) < n_total, (
            f"split_idx={split_idx} would exclude all {n_total} samples. "
            f"Use a value where |split_idx| < {n_total}."
        )

        idx = split_idx if exclude_end_set else None

        # ── Training slice ────────────────────────────────────────────────────
        self.temp_matrices_raw = data["temp_matrices"][:idx]
        self.spatial_masks     = data["masks"][:idx]            # (N, C, T)
        self.labels            = data["labels"][:idx]
        self.month_years       = data["month_years"][:idx]
        self.capitals          = data["capitals"].astype(str)

        # ── Holdout slice ─────────────────────────────────────────────────────
        if exclude_end_set:
            self._init_endset(data, split_idx)

        # ── Normalisation — training stats only, no leakage ───────────────────
        self.mu  = torch.tensor(self.temp_matrices_raw, dtype=torch.float32).mean(dim=0)
        self.std = torch.tensor(self.temp_matrices_raw, dtype=torch.float32).std(dim=0)
        self.temp_matrices = self._normalize(self.temp_matrices_raw, self.mu, self.std)

        # ── Temporal mask template (built once, broadcast per sample) ─────────
        T = self.temp_matrices.shape[-1]                        # e.g. 31
        assert observed_days <= T, (
            f"observed_days={observed_days} exceeds the time dimension T={T}."
        )
        self._temporal_template = self._make_temporal_template(T)

        print(
            f"[TemporalMaskDataset] {len(self.labels)} training samples | "
            f"Observing first {observed_days}/{T} days per month."
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_endset(self, data: dict, idx: int):
        """Slices the holdout arrays and logs the YEAR-Month range chosen."""
        self.temp_matrices_endset_raw = data["temp_matrices"][idx:]
        self.spatial_masks_endset     = data["masks"][idx:]
        self.labels_endset            = data["labels"][idx:]
        self.month_years_endset       = data["month_years"][idx:]

        first_my = self.month_years_endset[0]
        last_my  = self.month_years_endset[-1]
        print(
            f"[TemporalMaskDataset] Holdout end-set: {len(self.month_years_endset)} samples "
            f"| From {first_my} to {last_my}."
        )

    def _make_temporal_template(self, T: int) -> np.ndarray:
        """Returns a (1, T) float32 array: 1 for days < observed_days, 0 otherwise."""
        template = np.zeros((1, T), dtype=np.float32)
        template[0, : self.observed_days] = 1.0
        return template

    def _combine_masks(self, spatial_mask: np.ndarray) -> torch.Tensor:
        """
        Multiplies the per-station spatial mask (C, T) with the temporal
        template (1, T), broadcasting over channels. Returns (C, T) FloatTensor.
        """
        return torch.FloatTensor(spatial_mask * self._temporal_template)

    def get_which_month_reversed(self, idx: int) -> str:
        """Returns month_year at reverse position -idx (e.g. -1 → last month)."""
        return self.month_years[-idx]

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x    = torch.FloatTensor(self.temp_matrices[idx])      # (C, T)
        y    = torch.tensor([self.labels[idx]], dtype=torch.float32)
        mask = self._combine_masks(self.spatial_masks[idx])    # (C, T)
        return x, y, mask

    def get_month_year(self, idx: int) -> str:
        return self.month_years[idx]

    def get_endset_dataset(self) -> "TemporalMaskDS_EndSubset":
        """
        Returns the holdout subset normalised with training mu/std
        and carrying the same temporal cutoff template.
        """
        temp_matrices_endset = self._normalize(
            self.temp_matrices_endset_raw, self.mu, self.std
        )
        return TemporalMaskDS_EndSubset(
            temp_matrices=temp_matrices_endset,
            spatial_masks=self.spatial_masks_endset,
            labels=self.labels_endset,
            month_years=self.month_years_endset,
            temporal_template=self._temporal_template,
        )


class TemporalMaskDS_EndSubset(Dataset):
    """Holdout end-subset paired with TemporalMaskDataset."""

    def __init__(self, temp_matrices, spatial_masks, labels, month_years, temporal_template):
        self.temp_matrices      = temp_matrices
        self.spatial_masks      = spatial_masks
        self.labels             = labels
        self.month_years        = month_years
        self._temporal_template = temporal_template

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x    = torch.FloatTensor(self.temp_matrices[idx])
        y    = torch.tensor([self.labels[idx]], dtype=torch.float32)
        mask = torch.FloatTensor(self.spatial_masks[idx] * self._temporal_template)
        return x, y, mask

    def get_month_year(self, idx: int) -> str:
        return self.month_years[idx]


class TemporalMaskDataset_Finetun(BaseTempDataset):
    """
    Fine-tuning dataset for QuantileTempCNN using an external data source
    (e.g. Open-Meteo) normalised with pre-existing ERA5 mu/std.

    Key differences from TemporalMaskDataset:
      - No train/holdout split: the entire file is used for fine-tuning.
      - mu and std are injected externally (from the ERA5 training dataset)
        so the model sees the same input distribution it was pre-trained on.
      - Same temporal masking logic (observed_days) is preserved so the
        partial-month forecasting setup is identical at fine-tune time.

    Args:
        npz_path (str):       Path to the fine-tuning .npz archive.
        mu (Tensor):          Per-channel mean from ERA5 training set.
        std (Tensor):         Per-channel std  from ERA5 training set.
        observed_days (int):  Leading days to keep visible (default 20).
        return_mask (bool):   Return combined mask in __getitem__ (default True).

    Example:
        openmeteo_dataset = TemporalMaskDataset_Finetun(
            npz_path  = path_openmeteo_dataset,
            mu        = era5_train_ds.mu,
            std       = era5_train_ds.std,
            observed_days = 20,
        )
    """

    def __init__(
        self,
        npz_path: str,
        mu: torch.Tensor,
        std: torch.Tensor,
        observed_days: int = 20,
        return_mask: bool = True,
    ):
        # ── Assertions ────────────────────────────────────────────────────────
        assert isinstance(npz_path, str) and npz_path.endswith(".npz"), (
            f"npz_path must be a '.npz' file path, got: {npz_path!r}"
        )
        assert isinstance(observed_days, int) and observed_days > 0, (
            f"observed_days must be a positive int, got: {observed_days!r}"
        )
        assert isinstance(mu, torch.Tensor) and isinstance(std, torch.Tensor), (
            "mu and std must be torch.Tensors (use era5_dataset.mu / .std)."
        )

        self.observed_days = observed_days
        self.return_mask   = return_mask

        # ── Inject ERA5 normalisation stats (no leakage, no recomputation) ────
        self.mu  = mu
        self.std = std

        # ── Load full file — no split for fine-tuning ─────────────────────────
        data = np.load(npz_path, allow_pickle=True)

        self.temp_matrices_raw = data["temp_matrices"]          # (N, C, T)
        self.spatial_masks     = data["masks"]                  # (N, C, T)
        self.labels            = data["labels"]                 # (N,)
        self.month_years       = data["month_years"]            # (N,)
        self.capitals          = data["capitals"].astype(str)

        # ── Validate channel dimension matches ERA5 mu/std shape ──────────────
        C_data = self.temp_matrices_raw.shape[1]
        # mu is (C, T) when computed via .mean(dim=0) on a (N, C, T) tensor
        C_mu   = mu.shape[0]
        assert C_data == C_mu, (
            f"Channel mismatch: fine-tuning data has {C_data} channels "
            f"but ERA5 mu/std have {C_mu}. "
            "Ensure both datasets share the same capital/station set."
        )

        # ── Normalise with ERA5 stats ─────────────────────────────────────────
        self.temp_matrices = self._normalize(self.temp_matrices_raw, self.mu, self.std)

        # ── Temporal mask template ────────────────────────────────────────────
        T = self.temp_matrices.shape[-1]
        assert observed_days <= T, (
            f"observed_days={observed_days} exceeds the time dimension T={T}."
        )
        self._temporal_template = self._make_temporal_template(T)

        # ── Summary ───────────────────────────────────────────────────────────
        first_my = self.month_years[0]
        last_my  = self.month_years[-1]
        print(
            f"[TemporalMaskDataset_Finetun] {len(self.labels)} samples "
            f"| {first_my} → {last_my} "
            f"| Observing first {observed_days}/{T} days "
            f"| Normalised with ERA5 mu/std (shape: {tuple(mu.shape)})"
        )

    # ── Helpers (reuse from base + TemporalMaskDataset) ───────────────────────

    def _make_temporal_template(self, T: int) -> np.ndarray:
        template = np.zeros((1, T), dtype=np.float32)
        template[0, : self.observed_days] = 1.0
        return template

    def _combine_masks(self, spatial_mask: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(spatial_mask * self._temporal_template)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.FloatTensor(self.temp_matrices[idx])          # (C, T)
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        if self.return_mask:
            mask = self._combine_masks(self.spatial_masks[idx]) # (C, T)
            return x, y, mask
        return x, y

    def get_month_year(self, idx: int) -> str:
        return self.month_years[idx]