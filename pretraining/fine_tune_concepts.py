"""
Fine-tune pretrained encoder for concept prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ConceptDataset(Dataset):
    """Dataset for concept prediction with windowed sensor data."""
    
    def __init__(self, sensor_data: np.ndarray, concept_labels: np.ndarray, 
                 window_size: int = 128, stride: int = 32):
        """
        Initialize the concept dataset.
        
        Args:
            sensor_data: Raw sensor data (n_samples, 3)
            concept_labels: Concept labels (n_windows, 5) - 5 concepts
            window_size: Size of each time window
            stride: Stride for window creation
        """
        self.sensor_data = sensor_data
        self.concept_labels = concept_labels
        self.window_size = window_size
        self.stride = stride
        
        # Create windows
        self.windows = self._create_windows()
        
    def _create_windows(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create windows from sensor data and match with concept labels."""
        windows = []
        
        for i in range(0, len(self.sensor_data) - self.window_size + 1, self.stride):
            window = self.sensor_data[i:i + self.window_size]
            if len(window) == self.window_size:
                # Find corresponding concept label (approximate matching)
                concept_idx = min(i // self.stride, len(self.concept_labels) - 1)
                concept_label = self.concept_labels[concept_idx]
                windows.append((window, concept_label))
        
        return windows
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window, concept_label = self.windows[idx]
        return torch.FloatTensor(window.T), torch.FloatTensor(concept_label)  # Transpose for Conv1d


class PretrainedConceptPredictor(nn.Module):
    """Concept predictor using pretrained encoder."""
    
    def __init__(self, pretrained_encoder_path: str, input_dim: int = 3, 
                 hidden_dim: int = 64, num_concepts: int = 5, 
                 freeze_encoder: bool = True):
        """
        Initialize the concept predictor.
        
        Args:
            pretrained_encoder_path: Path to pretrained encoder weights
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_concepts: Number of concepts to predict (5)
            freeze_encoder: Whether to freeze pretrained encoder weights
        """
        super(PretrainedConceptPredictor, self).__init__()
        
        self.freeze_encoder = freeze_encoder
        
        # Load pretrained encoder
        self.encoder = self._load_pretrained_encoder(pretrained_encoder_path, input_dim, hidden_dim)
        
        if freeze_encoder:
            # Freeze encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Concept prediction head
        self.concept_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_concepts),
            nn.Sigmoid()  # Concepts are in [0,1] range
        )
        
    def _load_pretrained_encoder(self, path: str, input_dim: int, hidden_dim: int):
        """Load pretrained encoder architecture."""
        # Create encoder with same architecture as pretraining
        encoder = ImprovedTimeSeriesEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        
        # Load pretrained weights
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            # Filter out projection head weights
            encoder_state_dict = {}
            for key, value in checkpoint.items():
                if not key.startswith('projection.'):
                    encoder_state_dict[key] = value
            
            encoder.load_state_dict(encoder_state_dict)
            print(f"Loaded pretrained encoder from {path}")
        else:
            print(f"Warning: Pretrained encoder not found at {path}")
            
        return encoder
    
    def forward(self, x):
        """Forward pass through encoder and concept head."""
        # Get features from pretrained encoder
        features = self.encoder.get_features(x)
        
        # Predict concepts
        concepts = self.concept_head(features)
        
        return concepts


class ImprovedTimeSeriesEncoder(nn.Module):
    """Improved 1D-CNN Encoder (same as pretraining)."""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, 
                 num_layers: int = 3, kernel_size: int = 5, 
                 dropout: float = 0.2):
        super(ImprovedTimeSeriesEncoder, self).__init__()
        
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
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        """Forward pass through encoder."""
        x = self.encoder(x)
        x = self.global_pool(x)
        return x.squeeze(-1)
    
    def get_features(self, x):
        """Get features from encoder (for concept prediction)."""
        return self.forward(x)


def load_and_prepare_data():
    """Load and prepare data for concept prediction."""
    print("Loading data...")
    
    # Load raw sensor data
    raw_data = pd.read_csv('/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/rule_based_labeling/raw_with_features.csv')
    
    # Load windowed concept data
    window_data = pd.read_csv('/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/rule_based_labeling/window_with_features.csv')
    
    # Extract sensor data (x, y, z axes)
    sensor_data = raw_data[['x-axis', 'y-axis', 'z-axis']].values
    
    # Extract concept labels (5 concepts)
    concept_columns = ['periodicity', 'temporal_stability', 'coordination', 
                      'motion_intensity', 'vertical_dominance']
    concept_labels = window_data[concept_columns].values
    
    print(f"Sensor data shape: {sensor_data.shape}")
    print(f"Concept labels shape: {concept_labels.shape}")
    print(f"Concept columns: {concept_columns}")
    
    return sensor_data, concept_labels, concept_columns


def train_concept_predictor():
    """Train the concept predictor using pretrained encoder."""
    print("Starting concept prediction training...")
    
    # Load data
    sensor_data, concept_labels, concept_columns = load_and_prepare_data()
    
    # Load scaler from pretraining or create new one
    scaler_path = '/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining/improved_scaler.pkl'
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Loaded existing scaler from pretraining")
    except:
        print("Creating new scaler...")
        scaler = StandardScaler()
        scaler.fit(sensor_data)
        # Save the new scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print("New scaler created and saved")
    
    # Scale sensor data
    sensor_data_scaled = scaler.transform(sensor_data)
    
    # Create dataset
    dataset = ConceptDataset(sensor_data_scaled, concept_labels, window_size=128, stride=32)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = PretrainedConceptPredictor(
        pretrained_encoder_path='/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining/improved_pretrained_encoder.pth',
        input_dim=3,
        hidden_dim=64,
        num_concepts=5,
        freeze_encoder=True
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Calculate R² score
        val_predictions = np.concatenate(val_predictions)
        val_targets = np.concatenate(val_targets)
        r2_scores = []
        for i in range(5):  # 5 concepts
            r2 = r2_score(val_targets[:, i], val_predictions[:, i])
            r2_scores.append(r2)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  R² Scores: {[f"{r2:.3f}" for r2 in r2_scores]}')
        print(f'  Concepts: {concept_columns}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_concept_predictor.pth')
            print(f'  New best model saved!')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if epoch > 10 and val_loss > best_val_loss * 1.1:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_concept_predictor.pth'))
    
    # Final evaluation
    model.eval()
    final_predictions = []
    final_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            final_predictions.append(outputs.cpu().numpy())
            final_targets.append(batch_y.cpu().numpy())
    
    final_predictions = np.concatenate(final_predictions)
    final_targets = np.concatenate(final_targets)
    
    # Calculate final metrics
    print("\nFinal Results:")
    print("=" * 50)
    for i, concept in enumerate(concept_columns):
        mse = mean_squared_error(final_targets[:, i], final_predictions[:, i])
        r2 = r2_score(final_targets[:, i], final_predictions[:, i])
        print(f"{concept}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for i, concept in enumerate(concept_columns):
        r2 = r2_score(final_targets[:, i], final_predictions[:, i])
        plt.bar(concept, r2, alpha=0.7)
    plt.xlabel('Concepts')
    plt.ylabel('R² Score')
    plt.title('R² Score by Concept')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('concept_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nModel saved as 'best_concept_predictor.pth'")
    print(f"Results plot saved as 'concept_prediction_results.png'")


if __name__ == "__main__":
    train_concept_predictor()
