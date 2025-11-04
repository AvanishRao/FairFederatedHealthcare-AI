"""Baseline LSTM model for healthcare time-series prediction.

Implements a standard LSTM architecture for temporal health data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class BaselineLSTM(nn.Module):
    """LSTM model for time-series health data prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, output_size: int = 1, 
                 dropout: float = 0.3, bidirectional: bool = False):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer dimension
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout rate for regularization
            bidirectional: Whether to use bidirectional LSTM
        """
        super(BaselineLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Fully connected output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            hidden: Optional initial hidden state
            
        Returns:
            Tuple of (output, (hidden_state, cell_state))
        """
        # LSTM forward pass
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last time step output
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Final output layer
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, 
                        self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, 
                        self.hidden_size).to(device)
        return (h0, c0)


def train_step(model: BaselineLSTM, data: torch.Tensor, targets: torch.Tensor,
               criterion: nn.Module, optimizer: torch.optim.Optimizer,
               device: torch.device) -> float:
    """
    Single training step.
    
    Args:
        model: LSTM model
        data: Input data
        targets: Target labels
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        Loss value
    """
    model.train()
    optimizer.zero_grad()
    
    data, targets = data.to(device), targets.to(device)
    outputs, _ = model(data)
    
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model: BaselineLSTM, data_loader: torch.utils.data.DataLoader,
            criterion: nn.Module, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on validation/test data.
    
    Args:
        model: LSTM model
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (average_loss, predictions, targets)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs, _ = model(data)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    return avg_loss, predictions, targets
