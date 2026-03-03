import torch
import torch.nn as nn

class MaskTempCNN(nn.Module):
    """1D-CNN with Masking support (Standard version from your notebook)."""
    def __init__(self, in_channels, dropout):
        super().__init__()
        self.dropout = dropout

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout),
            
            nn.Conv1d(256, 512, kernel_size=5, padding='same'),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout),
            
            nn.Conv1d(512, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout),
            
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

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        
        x = self.conv_block(x)

        if mask is not None:
            mask_pooled = torch.mean(mask, dim=1, keepdim=True)
            valid_count = mask_pooled.sum(dim=2, keepdim=True).clamp(min=1)
            x = (x * mask_pooled).sum(dim=2) / valid_count.squeeze(-1)
        else:
            x = torch.mean(x, dim=2)
            
        return self.fc(x)

class YearMaskTempCNN(nn.Module):
    """1D-CNN with Masking and Scalar Year Feature Injection.
    
    Notice that the final linear layer now has an input size of 128 + 1 to accommodate the year scalar
    
    """
    def __init__(self, in_channels, dropout):
        super().__init__()
        self.dropout = dropout

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(256, 512, kernel_size=5, padding='same'),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(512, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(256, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout)
        )
        
        # Injection happens here: 128 (CNN features) + 1 (Year feature)
        self.fc = nn.Sequential(
            nn.Linear(128 + 1, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=self.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, year, mask=None):
        if mask is not None:
            x = x * mask
        
        x = self.conv_block(x)

        if mask is not None:
            mask_pooled = torch.mean(mask, dim=1, keepdim=True)
            valid_count = mask_pooled.sum(dim=2, keepdim=True).clamp(min=1)
            x_pooled = (x * mask_pooled).sum(dim=2) / valid_count.squeeze(-1)
        else:
            x_pooled = torch.mean(x, dim=2)
            
        # Concatenate the scalar year to the flattened CNN features
        x_combined = torch.cat((x_pooled, year), dim=1)
        return self.fc(x_combined)