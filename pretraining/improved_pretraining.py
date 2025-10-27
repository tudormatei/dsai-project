"""
Improved Self-Supervised Pretraining with Better Stability
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


class ImprovedSensorDataDataset(Dataset):
    """Improved dataset with better masking strategy."""
    
    def __init__(self, data: np.ndarray, window_size: int = 128, mask_ratio: float = 0.1, 
                 mask_length: int = 4, random_mask: bool = True):
        """
        Initialize the dataset with improved masking.
        
        Args:
            data: Sensor data array of shape (n_samples, n_features)
            window_size: Size of each time window
            mask_ratio: Ratio of timesteps to mask (reduced from 0.15 to 0.1)
            mask_length: Length of each mask segment (reduced from 8 to 4)
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
        for i in range(0, len(self.data) - self.window_size + 1, self.window_size // 4):  # More overlap
            window = self.data[i:i + self.window_size]
            if len(window) == self.window_size:
                windows.append(window)
        return windows
    
    def _create_mask(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create improved mask for the input window."""
        mask = np.zeros(len(window), dtype=bool)
        
        if self.random_mask:
            # Random masking with better distribution
            n_masks = max(1, int(len(window) * self.mask_ratio))
            mask_positions = np.random.choice(len(window), n_masks, replace=False)
            mask[mask_positions] = True
        else:
            # Fixed position masking
            n_masks = max(1, int(len(window) * self.mask_ratio))
            mask_positions = np.linspace(0, len(window) - 1, n_masks, dtype=int)
            mask[mask_positions] = True
        
        # Apply mask with mean imputation instead of zero
        masked_window = window.copy()
        mean_values = np.mean(window, axis=0)
        masked_window[mask] = mean_values  # Use mean instead of zero
        
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


class ImprovedTimeSeriesEncoder(nn.Module):
    """Improved 1D-CNN Encoder with better architecture."""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, 
                 num_layers: int = 3, kernel_size: int = 5, 
                 dropout: float = 0.2):
        """
        Initialize the improved encoder.
        
        Args:
            input_dim: Number of input features (3 for x, y, z axes)
            hidden_dim: Hidden dimension size (reduced from 128 to 64)
            num_layers: Number of CNN layers (reduced from 4 to 3)
            kernel_size: Kernel size for convolutions (increased from 3 to 5)
            dropout: Dropout rate (increased from 0.1 to 0.2)
        """
        super(ImprovedTimeSeriesEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Build encoder layers with better architecture
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
        
        # Improved projection head with more layers
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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


class ImprovedMaskReconstructTrainer:
    """Improved trainer with better stability."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the improved trainer.
        
        Args:
            model: The encoder model to train
            device: Device to run training on
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        
    def setup_optimizer(self, learning_rate: float = 5e-4, weight_decay: float = 1e-4):
        """Setup optimizer and scheduler with better parameters."""
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                   lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6)
    
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
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _calculate_masked_loss(self, reconstructed: torch.Tensor, 
                             target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculate loss only on masked positions with better stability."""
        # MSE loss on masked positions
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(reconstructed, target)
        
        # Apply mask - expand mask to match loss dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(loss)
        masked_loss = loss * mask_expanded.float()
        
        # Return mean loss over masked positions with stability
        num_masked = mask.sum().clamp(min=1)
        return masked_loss.sum() / num_masked
    
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
              epochs: int = 50, save_path: Optional[str] = None):
        """Train the model with improved stability."""
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model with early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Best model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return train_losses, val_losses


def load_and_preprocess_data(csv_path: str, window_size: int = 128) -> Tuple[np.ndarray, StandardScaler]:
    """
    Load and preprocess the sensor data with better normalization.
    
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
    
    # Better normalization with outlier handling
    scaler = StandardScaler()
    sensor_data = scaler.fit_transform(sensor_data)
    
    # Clip extreme outliers
    sensor_data = np.clip(sensor_data, -3, 3)
    
    print(f"Data shape: {sensor_data.shape}")
    print(f"Data range: [{sensor_data.min():.3f}, {sensor_data.max():.3f}]")
    
    return sensor_data, scaler


def create_data_loaders(sensor_data: np.ndarray, window_size: int = 128, 
                       batch_size: int = 32, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders with better parameters.
    
    Args:
        sensor_data: Preprocessed sensor data
        window_size: Size of time windows
        batch_size: Batch size for training (reduced from 64 to 32)
        test_size: Fraction of data to use for validation
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    # Split data
    train_data, val_data = train_test_split(sensor_data, test_size=test_size, random_state=42)
    
    # Create datasets with improved parameters
    train_dataset = ImprovedSensorDataDataset(train_data, window_size=window_size, mask_ratio=0.1)
    val_dataset = ImprovedSensorDataDataset(val_data, window_size=window_size, mask_ratio=0.1)
    
    # Create data loaders with reduced workers for stability
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def plot_training_curves(train_losses: List[float], val_losses: List[float], save_path: str = None):
    """Plot training curves."""
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Train Loss', alpha=0.8, linewidth=2)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Improved Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main training function with improved parameters."""
    # Improved configuration
    config = {
        'csv_path': '/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/dataset.csv',
        'window_size': 128,
        'batch_size': 32,          # Reduced from 64
        'epochs': 50,
        'learning_rate': 5e-4,     # Reduced from 1e-3
        'hidden_dim': 64,           # Reduced from 128
        'num_layers': 3,            # Reduced from 4
        'mask_ratio': 0.1,          # Reduced from 0.15
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
    
    # Create improved model
    model = ImprovedTimeSeriesEncoder(
        input_dim=3,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create improved trainer
    trainer = ImprovedMaskReconstructTrainer(model)
    trainer.setup_optimizer(learning_rate=config['learning_rate'])
    
    # Train model
    save_path = os.path.join(config['save_dir'], 'improved_pretrained_encoder.pth')
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, 
        epochs=config['epochs'], 
        save_path=save_path
    )
    
    # Plot results
    plot_path = os.path.join(config['save_dir'], 'improved_training_curves.png')
    plot_training_curves(train_losses, val_losses, plot_path)
    
    # Save scaler
    import joblib
    scaler_path = os.path.join(config['save_dir'], 'improved_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    print(f"\nImproved pretraining completed!")
    print(f"Best model saved to: {save_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Training curves saved to: {plot_path}")


if __name__ == "__main__":
    main()


