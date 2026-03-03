import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import mean_squared_error

try:
    import wandb
except ImportError:
    wandb = None

def set_seed(seed=42):
    """Ensures reproducibility for Monte Carlo runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def train_epoch(model, loader, criterion, optimizer, device, use_mask=True):
    model.train()
    losses, mse_scores = [], []
    for batch in loader:
        # Dynamic unpacking for Year vs No-Year datasets
        if len(batch) == 4:
            x, y, mask, year = [b.to(device) for b in batch]
        elif len(batch) == 3:
            x, y, mask = [b.to(device) for b in batch]
            year = None
        else:
            x, y = [b.to(device) for b in batch[:2]]
            mask, year = None, None

        optimizer.zero_grad()
        outputs = model(x, year, mask=mask) if year is not None else model(x, mask=mask)
        loss = criterion(outputs.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        mse_scores.append(mean_squared_error(y.cpu().numpy().flatten(), outputs.detach().cpu().numpy().flatten()))
    return np.mean(losses), np.mean(mse_scores)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, y_true, y_pred = 0, [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                x, y, mask, year = [b.to(device) for b in batch]
            elif len(batch) == 3:
                x, y, mask = [b.to(device) for b in batch]
                year = None
            else:
                x, y = [b.to(device) for b in batch[:2]]
                mask, year = None, None

            outputs = model(x, year, mask=mask) if year is not None else model(x, mask=mask)
            total_loss += criterion(outputs.squeeze(), y.squeeze()).item()
            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(outputs.cpu().numpy().flatten())
    
    return total_loss / len(loader), mean_squared_error(y_true, y_pred), (y_true, y_pred)

def train_model(model, train_loader, val_loader, config, device, checkpoint_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=config['patience'], verbose=True,
                                    delta=config['delta'])
    
    if config.get('use_wandb') and wandb is not None:
        wandb.init(project=config['project_wandb'],
                    config=config, ## save hyperparams
                    name=config['run_name'],
                    group = 'foundation',
                    config_exclude_keys = ['use_wandb', 'project_wandb', 'run_name','list_config_internals_fm', 'verbose'], ## exclude non-hyperparameter keys
                    rereinitinit = 'finish_previous'
                    )

    for epoch in range(1, config['epochs'] + 1):
        t_loss, t_mse = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_mse, _ = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}: Train Loss {t_loss:.4f} | Val Loss {v_loss:.4f}")
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: Train MSE {t_mse:.4f} | Val MSE {v_mse:.4f}")
        
        if config.get('use_wandb') and wandb is not None:
            wandb.log({"train_loss": t_loss,
                        "val_loss": v_loss,
                        "train_mse": t_mse,
                         "val_mse": v_mse, "epoch": epoch})

        early_stopping(v_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    ## finish wandb
    if config.get('use_wandb') and wandb is not None: 
        wandb.finish()
    return early_stopping.val_loss_min