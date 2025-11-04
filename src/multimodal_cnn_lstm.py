"""Multimodal CNN-LSTM model for healthcare data fusion.

Combines convolutional layers for feature extraction with LSTM for temporal modeling.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class MultimodalCNNLSTM(nn.Module):
    """Multimodal CNN-LSTM architecture for healthcare time-series with multiple modalities."""
    
    def __init__(self, temporal_input_size: int, static_input_size: int = 0,
                 cnn_channels: list = [32, 64, 128], 
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.3):
        """
        Initialize multimodal CNN-LSTM model.
        
        Args:
            temporal_input_size: Number of temporal input features
            static_input_size: Number of static input features (demographics, etc.)
            cnn_channels: List of CNN channel sizes for each layer
            lstm_hidden_size: Hidden size for LSTM
            lstm_num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout rate
        """
        super(MultimodalCNNLSTM, self).__init__()
        
        self.temporal_input_size = temporal_input_size
        self.static_input_size = static_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # CNN layers for feature extraction from temporal data
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels in cnn_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels
        
        # LSTM layer for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Fusion layer combining LSTM output and static features
        fusion_input_size = lstm_hidden_size + static_input_size
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output_layer = nn.Linear(64, output_size)
        
    def forward(self, temporal_data: torch.Tensor, 
                static_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            temporal_data: Temporal input of shape (batch_size, seq_length, features)
            static_data: Static input of shape (batch_size, static_features)
            
        Returns:
            Output predictions
        """
        batch_size, seq_len, features = temporal_data.shape
        
        # Reshape for CNN: (batch, channels, length)
        x = temporal_data.permute(0, 2, 1)
        
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Reshape back for LSTM: (batch, seq_length, features)
        x = x.permute(0, 2, 1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output
        lstm_out = lstm_out[:, -1, :]
        
        # Fuse with static features if available
        if static_data is not None and self.static_input_size > 0:
            fused = torch.cat([lstm_out, static_data], dim=1)
        else:
            fused = lstm_out
        
        # Apply fusion layers
        fused = self.fusion_layers(fused)
        
        # Output layer
        output = self.output_layer(fused)
        
        return output


class AttentionMultimodalCNNLSTM(MultimodalCNNLSTM):
    """Enhanced multimodal CNN-LSTM with attention mechanism."""
    
    def __init__(self, *args, **kwargs):
        super(AttentionMultimodalCNNLSTM, self).__init__(*args, **kwargs)
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, temporal_data: torch.Tensor, 
                static_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with attention mechanism.
        
        Args:
            temporal_data: Temporal input of shape (batch_size, seq_length, features)
            static_data: Static input of shape (batch_size, static_features)
            
        Returns:
            Output predictions
        """
        batch_size, seq_len, features = temporal_data.shape
        
        # Reshape for CNN
        x = temporal_data.permute(0, 2, 1)
        
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)
        
        # Apply LSTM (get all time steps)
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fuse with static features
        if static_data is not None and self.static_input_size > 0:
            fused = torch.cat([attended_output, static_data], dim=1)
        else:
            fused = attended_output
        
        # Apply fusion and output layers
        fused = self.fusion_layers(fused)
        output = self.output_layer(fused)
        
        return output
