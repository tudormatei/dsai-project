"""
Self-Supervised Pretraining for Time-Series Sensor Data
Mask-and-Reconstruct Task Implementation

This module implements a self-supervised learning approach for pretraining
a 1D-CNN encoder on unlabeled sensor data using the mask-and-reconstruct task.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class SensorDataDataset(Dataset):
    """Dataset class for sensor data with windowing and masking."""
    
    def __init__(self, data: np.ndarray, window_size: int = 128, mask_ratio: float = 0.15, 
                 mask_length: int = 8, random_mask: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data: Sensor data array of shape (n_samples, n_features)
            window_size: Size of each time window
            mask_ratio: Ratio of timesteps to mask
            mask_length: Length of each mask segment
            random_mask: Whether to use random masking or fixed positions
        """
        self.data = data
        self.window_size = window_size
        self.mask_ratio = mask_ratio
        self.mask_length = mask_length
        self.random_mask = random_mask
        
        # Create windows
        self.windows = self._create_windows()
        
    def _create_windows(self) -> List[np.ndarray]:
        """Create sliding windows from the sensor data."""
        windows = []
        for i in range(0, len(self.data) - self.window_size + 1, self.window_size // 2):
            window = self.data[i:i + self.window_size]
            if len(window) == self.window_size:
                windows.append(window)
        return windows
    
    def _create_mask(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create mask for the input window."""
        mask = np.zeros(len(window), dtype=bool)
        
        if self.random_mask:
            # Random masking
            n_masks = int(len(window) * self.mask_ratio)
            mask_positions = np.random.choice(len(window), n_masks, replace=False)
            mask[mask_positions] = True
        else:
            # Fixed position masking
            n_masks = int(len(window) * self.mask_ratio)
            mask_positions = np.linspace(0, len(window) - 1, n_masks, dtype=int)
            mask[mask_positions] = True
        
        # Apply mask
        masked_window = window.copy()
        masked_window[mask] = 0  # Simple zero masking
        
        return masked_window, mask
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        masked_window, mask = self._create_mask(window)
        
        return {
            'input': torch.FloatTensor(masked_window),
            'target': torch.FloatTensor(window),
            'mask': torch.BoolTensor(mask)
        }


class TimeSeriesEncoder(nn.Module):
    """1D-CNN Encoder for time-series sensor data."""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, 
                 num_layers: int = 4, kernel_size: int = 3, 
                 dropout: float = 0.1):
        """
        Initialize the encoder.
        
        Args:
            input_dim: Number of input features (3 for x, y, z axes)
            hidden_dim: Hidden dimension size
            num_layers: Number of CNN layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
        """
        super(TimeSeriesEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Build encoder layers
        layers = []
        in_channels = input_dim
        
        for i in range(num_layers):
            out_channels = hidden_dim // (2 ** (num_layers - 1 - i))
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head for reconstruction
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            encoded: Encoded representation
            reconstructed: Reconstructed input
        """
        # Reshape for Conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Encode
        encoded = self.encoder(x)
        
        # Global pooling
        pooled = self.global_pool(encoded).squeeze(-1)
        
        # Reconstruct - need to expand to match input sequence length
        seq_len = x.shape[2]
        reconstructed = self.projection(pooled)
        reconstructed = reconstructed.unsqueeze(1).expand(-1, seq_len, -1)
        
        return encoded, reconstructed


class MaskReconstructTrainer:
    """Trainer class for mask-and-reconstruct pretraining."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model: The encoder model to train
            device: Device to run training on
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        
    def setup_optimizer(self, learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            input_data = batch['input'].to(self.device)
            target_data = batch['target'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            encoded, reconstructed = self.model(input_data)
            
            # Calculate loss only on masked positions
            loss = self._calculate_masked_loss(reconstructed, target_data, mask)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _calculate_masked_loss(self, reconstructed: torch.Tensor, 
                             target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculate loss only on masked positions."""
        # MSE loss on masked positions
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(reconstructed, target)
        
        # Apply mask - expand mask to match loss dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(loss)
        masked_loss = loss * mask_expanded.float()
        
        # Return mean loss over masked positions
        return masked_loss.sum() / mask.sum().clamp(min=1)
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_data = batch['input'].to(self.device)
                target_data = batch['target'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                encoded, reconstructed = self.model(input_data)
                loss = self._calculate_masked_loss(reconstructed, target_data, mask)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, save_path: Optional[str] = None):
        """Train the model."""
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Best model saved to {save_path}")
        
        return train_losses, val_losses


def load_and_preprocess_data(csv_path: str, window_size: int = 128) -> Tuple[np.ndarray, StandardScaler]:
    """
    Load and preprocess the sensor data.
    
    Args:
        csv_path: Path to the CSV file
        window_size: Size of time windows
    
    Returns:
        processed_data: Preprocessed sensor data
        scaler: Fitted scaler for inverse transformation
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Extract sensor data (x, y, z axes)
    sensor_data = df[['x-axis', 'y-axis', 'z-axis']].values
    
    # Normalize the data
    scaler = StandardScaler()
    sensor_data = scaler.fit_transform(sensor_data)
    
    print(f"Data shape: {sensor_data.shape}")
    print(f"Data range: [{sensor_data.min():.3f}, {sensor_data.max():.3f}]")
    
    return sensor_data, scaler


def create_data_loaders(sensor_data: np.ndarray, window_size: int = 128, 
                       batch_size: int = 32, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        sensor_data: Preprocessed sensor data
        window_size: Size of time windows
        batch_size: Batch size for training
        test_size: Fraction of data to use for validation
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    # Split data
    train_data, val_data = train_test_split(sensor_data, test_size=test_size, random_state=42)
    
    # Create datasets
    train_dataset = SensorDataDataset(train_data, window_size=window_size)
    val_dataset = SensorDataDataset(val_data, window_size=window_size)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def plot_training_curves(train_losses: List[float], val_losses: List[float], save_path: str = None):
    """Plot training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main training function."""
    # Configuration
    config = {
        'csv_path': '/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/dataset.csv',
        'window_size': 128,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 1e-3,
        'hidden_dim': 128,
        'num_layers': 4,
        'mask_ratio': 0.15,
        'save_dir': '/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining'
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load and preprocess data
    sensor_data, scaler = load_and_preprocess_data(config['csv_path'], config['window_size'])
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        sensor_data, 
        window_size=config['window_size'],
        batch_size=config['batch_size']
    )
    
    # Create model
    model = TimeSeriesEncoder(
        input_dim=3,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = MaskReconstructTrainer(model)
    trainer.setup_optimizer(learning_rate=config['learning_rate'])
    
    # Train model
    save_path = os.path.join(config['save_dir'], 'pretrained_encoder.pth')
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, 
        epochs=config['epochs'], 
        save_path=save_path
    )
    
    # Plot results
    plot_path = os.path.join(config['save_dir'], 'training_curves.png')
    plot_training_curves(train_losses, val_losses, plot_path)
    
    # Save scaler
    import joblib
    scaler_path = os.path.join(config['save_dir'], 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    print(f"\nPretraining completed!")
    print(f"Best model saved to: {save_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Training curves saved to: {plot_path}")


if __name__ == "__main__":
    main()
