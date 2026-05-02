import torch
import numpy as np
from sklearn.metrics import mean_pinball_loss

try:
    import wandb
except ImportError:
    wandb = None

from utils import set_seed, EarlyStopping          # reuse your existing helpers
from quantile_model import PinballLoss, QUANTILES


# ── Epoch helpers ─────────────────────────────────────────────────────────────

def train_epoch_quantile(model, loader, criterion, optimizer, device):
    """
    One training pass for the quantile model.

    Batch shape from TemporalMaskDataset: (x, y, mask)  – no year.
    Returns mean pinball loss and mean sklearn pinball score (q=0.50 as proxy).
    """
    model.train()
    losses, pinball_scores = [], []

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        optimizer.zero_grad()
        preds = model(x, mask=mask)             # (B, Q)
        loss  = criterion(preds, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # sklearn's mean_pinball_loss at the median (q=0.50) as a readable metric
        y_np    = y.cpu().numpy().flatten()
        pred_np = preds.detach().cpu().numpy()
        median_idx = len(QUANTILES) // 2        # index of q=0.50
        pinball_scores.append(
            mean_pinball_loss(y_np, pred_np[:, median_idx], alpha=0.50)
        )

    return np.mean(losses), np.mean(pinball_scores)


def evaluate_model_quantile(model, loader, criterion, device):
    """
    Validation / test pass.

    Returns:
        avg_loss        – mean pinball loss over all quantiles
        median_pinball  – sklearn pinball at q=0.50 (interpretable proxy)
        (y_true, y_pred_all) – arrays for post-hoc analysis
            y_pred_all shape: (N, Q)
    """
    model.eval()
    total_loss = 0.0
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            preds = model(x, mask=mask)         # (B, Q)
            total_loss += criterion(preds, y).item()

            y_true_all.extend(y.cpu().numpy().flatten())
            y_pred_all.append(preds.cpu().numpy())

    y_true_arr = np.array(y_true_all)           # (N,)
    y_pred_arr = np.vstack(y_pred_all)           # (N, Q)

    median_idx    = len(QUANTILES) // 2
    median_pinball = mean_pinball_loss(
        y_true_arr, y_pred_arr[:, median_idx], alpha=0.50
    )

    avg_loss = total_loss / len(loader)
    return avg_loss, median_pinball, (y_true_arr, y_pred_arr)


# ── Main training function ────────────────────────────────────────────────────

def train_quantile_model(model, train_loader, val_loader, config, device, checkpoint_path):
    """
    Trains QuantileTempCNN with PinballLoss.

    config keys (same convention as your existing train_model):
        lr, patience, verbose, delta_early_stop, epochs
        use_wandb, project_wandb, run_name   (optional)

    Returns:
        best validation pinball loss (scalar)
    """
    criterion = PinballLoss(quantiles=QUANTILES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    early_stopping = EarlyStopping(
        patience=config["patience"],
        verbose=config["verbose"],
        delta=config["delta_early_stop"],
    )

    if config.get("use_wandb") and wandb is not None:
        wandb.init(
            project=config["project_wandb"],
            config=config,
            name=config["run_name"],
            group="quantile",
            config_exclude_keys=["use_wandb", "project_wandb", "run_name", "verbose"],
            reinit="finish_previous",
        )

    for epoch in range(1, config["epochs"] + 1):
        t_loss, t_pinball = train_epoch_quantile(model, train_loader, criterion, optimizer, device)
        v_loss, v_pinball, _ = evaluate_model_quantile(model, val_loader, criterion, device)

        if config["verbose"] and (epoch % 10 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:>4} | "
                f"Train PB {t_loss:.4f}  median↓{t_pinball:.4f} | "
                f"Val PB {v_loss:.4f}  median↓{v_pinball:.4f}"
            )

        if config.get("use_wandb") and wandb is not None:
            wandb.log({
                "train_pinball_loss": t_loss,
                "train_median_pinball": t_pinball,
                "val_pinball_loss": v_loss,
                "val_median_pinball": v_pinball,
                "epoch": epoch,
            })

        early_stopping(v_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    if config.get("use_wandb") and wandb is not None:
        wandb.finish()

    return early_stopping.val_loss_min