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

    def forward(self, x,year=None, mask=None):
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
    
# class YearMaskTempCNN(nn.Module):
#     """1D-CNN with Masking and Scalar Year Feature Injection.
    
#     Notice that the final linear layer now has an input size of 128 + 1 to accommodate the year scalar

#     """
#     def __init__(self, in_channels, dropout):
#         super().__init__()
#         self.dropout = dropout

#         self.conv_block = nn.Sequential(
#             nn.Conv1d(in_channels, 256, kernel_size=5, padding='same'),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(p=self.dropout),
#             nn.Conv1d(256, 512, kernel_size=5, padding='same'),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(p=self.dropout),
#             nn.Conv1d(512, 256, kernel_size=3, padding='same'),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(p=self.dropout),
#             nn.Conv1d(256, 128, kernel_size=3, padding='same'),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(p=self.dropout)
#         )
        
#         # Injection happens here: 128 (CNN features) + 1 (Year feature)
#         self.fc = nn.Sequential(
#             nn.Linear(128 + 1, 64),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(p=self.dropout),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x, year, mask=None):
#         if mask is not None:
#             x = x * mask
        
#         x = self.conv_block(x)

#         if mask is not None:
#             mask_pooled = torch.mean(mask, dim=1, keepdim=True)
#             valid_count = mask_pooled.sum(dim=2, keepdim=True).clamp(min=1)
#             x_pooled = (x * mask_pooled).sum(dim=2) / valid_count.squeeze(-1)
#         else:
#             x_pooled = torch.mean(x, dim=2)
            
#         # Concatenate the scalar year to the flattened CNN features
#         x_combined = torch.cat((x_pooled, year), dim=1)
#         return self.fc(x_combined)