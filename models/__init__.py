# Models package - exportable classes and functions

# Architectures
from .architectures import (MaskTempCNN, YearMaskTempCNN, Detrend_MaskTempCNN, DetrendLoaderWrapper)

# Data loaders
from .data_loader import (
    BaseTempDataset,
    YearTempDSMask,
    YearTempDSMask_Finetun,
    TempDS2025
    
)

# Training utilities
from .trainer import (set_seed, EarlyStopping, 
                                train_epoch,
                                evaluate_model,
                                train_model
                                )

__all__ = [
    # Architectures
    "MaskTempCNN",
    "YearMaskTempCNN",
    # Data loaders
    "BaseTempDataset",
    "YearTempDSMask",
    "TempDS2025",
    "YearTempDSMask_Finetun",
    # Training
    "set_seed",
    "EarlyStopping",
    "train_epoch",
    "train_model",
    "evaluate_model",
    "Detrend_MaskTempCNN",
    "DetrendLoaderWrapper"
]
