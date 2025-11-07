"""
Fine-tune pretrained encoder for concept prediction.

Pipeline overview:
1. Load sensor windows and concept labels from the submission dataset.
2. Preprocess data: robust window extraction, scaling, and deterministic splitting.
3. Optionally augment training windows (jitter, scaling, rotation) with rare-class boosting.
4. Fine-tune the pretrained encoder with multi-head classification heads.
5. Evaluate using accuracy, weighted F1, AUROC, and confusion matrices.
6. Persist the best PyTorch weights and generate diagnostic plots.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pickle
from typing import Tuple, List, Optional
import warnings
import random
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
# Fixed seeds ensure fair comparison between augmentation and no augmentation
# Without seeds, you'd need multiple runs to see the true effect due to variance
RANDOM_SEED = 42
USE_FIXED_SEED = True  # Set to False for non-deterministic runs (higher variance)

if USE_FIXED_SEED:
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Using fixed random seed for reproducibility")
else:
    print("Using non-deterministic random initialization (results will vary)")

# Paths relative to the new submission directory structure
CURRENT_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = CURRENT_DIR.parent
DATA_DIR = SUBMISSION_DIR / 'data'


class ConceptDataset(Dataset):
    """Dataset for concept prediction with windowed sensor data."""
    
    def __init__(self, windows: List[np.ndarray], concept_labels: np.ndarray):
        """
        Initialize the concept dataset.
        
        Args:
            windows: List of window arrays, each of shape (window_size, 3)
            concept_labels: Concept labels (n_windows, 5) - 5 concepts
        """
        self.windows = windows
        self.concept_labels = concept_labels
        
        assert len(windows) == len(concept_labels), \
            f"Mismatch: {len(windows)} windows but {len(concept_labels)} labels"
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        concept_label = self.concept_labels[idx]
        # Transpose for Conv1d: (window_size, 3) -> (3, window_size)
        # Convert labels from [0.0, 0.5, 1.0] to class indices [0, 1, 2]
        class_indices = (concept_label * 2).astype(int)
        return torch.FloatTensor(window.T), torch.LongTensor(class_indices)


class PretrainedConceptPredictor(nn.Module):
    """Concept predictor using pretrained encoder for multi-class classification."""
    
    def __init__(self, pretrained_encoder_path: str, input_dim: int = 3, 
                 hidden_dim: int = 64, num_concepts: int = 5, num_classes: int = 3,
                 freeze_encoder: bool = True):
        """
        Initialize the concept predictor.
        
        Args:
            pretrained_encoder_path: Path to pretrained encoder weights
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_concepts: Number of concepts to predict (5)
            num_classes: Number of classes per concept (3: 0.0, 0.5, 1.0)
            freeze_encoder: Whether to freeze pretrained encoder weights
        """
        super(PretrainedConceptPredictor, self).__init__()
        
        self.freeze_encoder = freeze_encoder
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        
        # Load pretrained encoder
        self.encoder = self._load_pretrained_encoder(pretrained_encoder_path, input_dim, hidden_dim)
        
        if freeze_encoder:
            # Freeze encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Shared feature processing
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.concept_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 4, num_classes) for _ in range(num_concepts)
        ])
        
    def _load_pretrained_encoder(self, path: str, input_dim: int, hidden_dim: int):
        """Load pretrained encoder architecture."""
        # Create encoder with same architecture as pretraining
        encoder = ImprovedTimeSeriesEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        
        # Load pretrained weights
        path = Path(path)
        if path.exists():
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
        """Forward pass through encoder and concept heads."""
        # Get features from pretrained encoder
        features = self.encoder.get_features(x)
        
        # Shared processing
        shared_features = self.shared_head(features)
        
        # Predict concepts (each head outputs logits for 3 classes)
        concept_outputs = [head(shared_features) for head in self.concept_heads]
        
        return concept_outputs


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


def extract_window_robust(df_sensor, window_row, time_tolerance=0.5, target_length=60):
    """
    Extract sensor data window with time tolerance to handle mismatches.
    Matches the approach used in the training notebooks.
    """
    user = window_row['user']
    activity = window_row['activity']
    start_time = window_row['start_time']
    end_time = window_row['end_time']
    
    # Get data for this user/activity
    user_activity_data = df_sensor[(df_sensor['user'] == user) & 
                                  (df_sensor['activity'] == activity)].copy()
    
    if len(user_activity_data) == 0:
        return None
    
    # Find data within time window with tolerance
    mask = ((user_activity_data['time_s'] >= start_time - time_tolerance) & 
            (user_activity_data['time_s'] <= end_time + time_tolerance))
    
    window_data = user_activity_data[mask]
    
    if len(window_data) < 10:  # Need minimum samples
        return None
    
    # Extract sensor readings
    sensor_data = window_data[['x-axis', 'y-axis', 'z-axis']].values
    
    # Pad or truncate to fixed length
    if len(sensor_data) > target_length:
        # Randomly sample if too long
        indices = np.random.choice(len(sensor_data), target_length, replace=False)
        sensor_data = sensor_data[indices]
    elif len(sensor_data) < target_length:
        # Pad with last value if too short
        padding = np.tile(sensor_data[-1:], (target_length - len(sensor_data), 1))
        sensor_data = np.vstack([sensor_data, padding])
    
    return sensor_data


def load_and_prepare_data():
    """Load and prepare data for concept prediction using submission data."""
    print("Loading data from submission/data/...")

    # Load sensor data
    sensor_path = DATA_DIR / 'final_dataset.csv'
    df_sensor = pd.read_csv(sensor_path)

    # Load window labels
    windows_path = DATA_DIR / 'final_window_labels.csv'
    df_windows = pd.read_csv(windows_path)

    print(f"Sensor data: {len(df_sensor)} readings")
    print(f"Window labels: {len(df_windows)} windows")
    
    # Extract concept labels (5 concepts) - using the correct concepts
    concept_columns = ['periodicity', 'temporal_stability', 'coordination', 
                      'movement_variability', 'movement_consistency']
    
    # Verify all concept columns exist
    missing_cols = [col for col in concept_columns if col not in df_windows.columns]
    if missing_cols:
        raise ValueError(f"Missing concept columns in window labels: {missing_cols}")
    
    concept_labels = df_windows[concept_columns].values
    
    # Extract windows from sensor data
    print("Extracting windows from sensor data...")
    windows = []
    valid_indices = []
    
    for idx, (_, window_row) in enumerate(df_windows.iterrows()):
        window_data = extract_window_robust(df_sensor, window_row, target_length=60)
        if window_data is not None:
            windows.append(window_data)
            valid_indices.append(idx)
        else:
            print(f"Warning: Failed to extract window {idx}")
    
    # Filter concept labels to match valid windows
    concept_labels = concept_labels[valid_indices]
    
    print(f"Successfully extracted {len(windows)} windows")
    print(f"Concept labels shape: {concept_labels.shape}")
    print(f"Concept columns: {concept_columns}")
    print(f"\nConcept value ranges:")
    for i, concept in enumerate(concept_columns):
        print(f"  {concept}: [{concept_labels[:, i].min():.2f}, {concept_labels[:, i].max():.2f}]")
    
    return windows, concept_labels, concept_columns


def augment_jitter(data, sigma=0.05):
    """Add additive Gaussian noise to simulate sensor imperfections.

    Args:
        data: Array shaped (timesteps, 3) representing one window.
        sigma: Standard deviation of the zero-mean Gaussian noise.

    Returns:
        Augmented window with jitter applied per axis at every timestep.
    """
    # data shape: (timesteps, 3)
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise


def augment_scaling(data, sigma=0.1):
    """Scale sensor magnitude to emulate different movement intensities.

    Args:
        data: Array shaped (timesteps, 3) representing one window.
        sigma: Standard deviation of the multiplicative scale noise.

    Returns:
        Window scaled independently per axis but consistently across time.
    """
    # data shape: (timesteps, 3)
    # Apply same scaling factor to all timesteps but different for each axis
    scale_factors = np.random.normal(1.0, sigma, (1, 3))
    return data * scale_factors


def augment_rotation(data):
    """Rotate the window around the z-axis to mimic orientation changes.

    Args:
        data: Array shaped (timesteps, 3) representing one window.

    Returns:
        Window rotated by a random angle sampled from ±30 degrees.
    """
    # data shape: (timesteps, 3)
    angle = np.random.uniform(-np.pi/6, np.pi/6)  # ±30 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    # Apply rotation to each timestep
    return np.dot(data, rotation_matrix.T)


def augment_training_data(windows, concept_labels, concept_columns, base_factor=10, rare_class_boost=3):
    """
    Augment training data with special focus on underrepresented classes.
    
    Args:
        windows: List of window arrays (each is (timesteps, 3))
        concept_labels: Array of concept labels (n_samples, n_concepts)
        concept_columns: List of concept names
        base_factor: Base augmentation multiplier
        rare_class_boost: Additional multiplier for rare classes
    
    Returns:
        Augmented windows and labels
    """
    print("\n" + "=" * 60)
    print("APPLYING DATA AUGMENTATION")
    print("=" * 60)
    
    n_original = len(windows)
    class_indices = (concept_labels * 2).astype(int)
    
    # Analyze class distribution to identify rare classes
    print("\nAnalyzing class distribution for augmentation...")
    class_counts = {}
    for i, concept in enumerate(concept_columns):
        unique, counts = np.unique(class_indices[:, i], return_counts=True)
        class_counts[concept] = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}
        print(f"  {concept}: {class_counts[concept]}")
    
    # Find rare classes (classes with < 20% of max class count)
    rare_classes = {}
    for i, concept in enumerate(concept_columns):
        counts = class_counts[concept]
        if len(counts) > 0:
            max_count = max(counts.values())
            rare_classes[concept] = [
                cls for cls, cnt in counts.items() 
                if cnt < max_count * 0.2
            ]
    
    print(f"\nRare classes identified: {rare_classes}")
    
    # Convert windows to numpy array for easier manipulation
    windows_array = np.array([w for w in windows])  # (n_samples, timesteps, 3)
    
    # Create augmented data
    augmented_windows = [windows_array]
    augmented_labels = [concept_labels]
    
    # Base augmentation for all samples
    print(f"\nApplying base augmentation (factor={base_factor})...")
    for aug_iter in range(base_factor):
        aug_windows = []
        aug_labels = []
        
        # Set seed for this augmentation iteration to ensure reproducibility
        np.random.seed(RANDOM_SEED + aug_iter)
        
        for idx in range(n_original):
            # Randomly select augmentation type
            aug_type = np.random.choice(['jitter', 'scaling', 'rotation', 'none'])
            
            window = windows_array[idx]
            if aug_type == 'jitter':
                aug_window = augment_jitter(window, sigma=0.05)
            elif aug_type == 'scaling':
                aug_window = augment_scaling(window, sigma=0.1)
            elif aug_type == 'rotation':
                aug_window = augment_rotation(window)
            else:  # none
                aug_window = window.copy()
            
            aug_windows.append(aug_window)
            aug_labels.append(concept_labels[idx])
        
        augmented_windows.append(np.array(aug_windows))
        augmented_labels.append(np.array(aug_labels))
    
    # Additional augmentation for rare classes
    if rare_class_boost > 0:
        print(f"\nApplying additional augmentation for rare classes (boost={rare_class_boost})...")
        for i, concept in enumerate(concept_columns):
            if concept in rare_classes and len(rare_classes[concept]) > 0:
                rare_cls = rare_classes[concept]
                # Find indices of samples with rare classes for this concept
                rare_indices = [
                    idx for idx in range(n_original)
                    if class_indices[idx, i] in rare_cls
                ]
                
                if len(rare_indices) > 0:
                    print(f"  {concept}: Augmenting {len(rare_indices)} rare class samples")
                    for boost_iter in range(rare_class_boost):
                        aug_windows = []
                        aug_labels = []
                        
                        # Set seed for this boost iteration
                        np.random.seed(RANDOM_SEED + 1000 + i * 100 + boost_iter)
                        
                        for idx in rare_indices:
                            # Randomly select augmentation type
                            aug_type = np.random.choice(['jitter', 'scaling', 'rotation'])
                            
                            window = windows_array[idx]
                            if aug_type == 'jitter':
                                aug_window = augment_jitter(window, sigma=0.05)
                            elif aug_type == 'scaling':
                                aug_window = augment_scaling(window, sigma=0.1)
                            else:  # rotation
                                aug_window = augment_rotation(window)
                            
                            aug_windows.append(aug_window)
                            aug_labels.append(concept_labels[idx])
                        
                        augmented_windows.append(np.array(aug_windows))
                        augmented_labels.append(np.array(aug_labels))
    
    # Combine all augmented data
    final_windows = np.concatenate(augmented_windows, axis=0)
    final_labels = np.concatenate(augmented_labels, axis=0)
    
    # Convert back to list format (for compatibility with existing code)
    final_windows_list = [final_windows[i] for i in range(len(final_windows))]
    
    augmentation_factor = len(final_windows) / n_original
    print(f"\nAugmentation complete!")
    print(f"   Original samples: {n_original}")
    print(f"   Augmented samples: {len(final_windows)}")
    print(f"   Augmentation factor: {augmentation_factor:.2f}x")
    print("=" * 60)
    
    return final_windows_list, final_labels


def create_stratified_split(windows, concept_labels, concept_columns, test_size=0.2, max_attempts=100):
    """Create a robust stratified split for multi-concept classification.

    The function stratifies on the first concept (periodicity) and then validates
    that every concept retains all three classes in both subsets. If any concept
    is missing a class, it retries with a different random seed until either a
    valid split is found or ``max_attempts`` is reached.

    Args:
        windows: Sequence of scaled sensor windows.
        concept_labels: Array of shape (n_samples, 5) with numeric labels.
        concept_columns: Ordered list of concept names.
        test_size: Proportion reserved for validation.
        max_attempts: Maximum number of retries before giving up.

    Returns:
        Tuple of (train_indices, val_indices) guaranteeing class coverage.

    Raises:
        ValueError: If no valid split can be produced within ``max_attempts``.
    """
    print("\n" + "=" * 60)
    print("CREATING STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 60)
    
    # Convert concept labels to class indices (0, 1, 2)
    class_indices = (concept_labels * 2).astype(int)
    n_samples = len(windows)
    n_concepts = len(concept_columns)
    
    print(f"Total samples: {n_samples}")
    print(f"Number of concepts: {n_concepts}")
    
    # Check class distribution for each concept
    print("\nClass distribution (before split):")
    print("-" * 60)
    for i, concept in enumerate(concept_columns):
        unique, counts = np.unique(class_indices[:, i], return_counts=True)
        print(f"{concept:25s}: ", end="")
        for cls, cnt in zip(unique, counts):
            print(f"Class {cls}: {cnt:4d}  ", end="")
        print()
    
    # Try multiple splits until we get one where all classes are present in both sets
    for attempt in range(max_attempts):
        # Create a combined stratification label
        # Use the first concept for stratification (or combine multiple concepts)
        # For simplicity, we'll stratify on the first concept and validate all
        stratify_labels = class_indices[:, 0]  
        
        try:
            # Create stratified split
            indices = np.arange(n_samples)
            train_idx, val_idx = train_test_split(
                indices,
                test_size=test_size,
                stratify=stratify_labels,
                random_state=42 + attempt  # Vary random state on each attemp
            )
        except ValueError as e:
            # If stratification fails (e.g., not enough samples per class), use random split
            print(f"⚠️  Stratification failed on attempt {attempt + 1}: {e}")
            print("   Falling back to random split...")
            indices = np.arange(n_samples)
            train_idx, val_idx = train_test_split(
                indices,
                test_size=test_size,
                random_state=42 + attempt
            )
        
        # Validate that all classes are present in both train and validation for ALL concepts
        all_valid = True
        missing_classes = []
        
        for i, concept in enumerate(concept_columns):
            train_classes = set(class_indices[train_idx, i])
            val_classes = set(class_indices[val_idx, i])
            all_classes = set([0, 1, 2])  # Expected classes
            
            train_missing = all_classes - train_classes
            val_missing = all_classes - val_classes
            
            if train_missing or val_missing:
                all_valid = False
                missing_classes.append({
                    'concept': concept,
                    'train_missing': train_missing,
                    'val_missing': val_missing
                })
        
        if all_valid:
            print(f"\n✅ Valid split found on attempt {attempt + 1}!")
            print("\nClass distribution (after split):")
            print("-" * 60)
            print(f"{'Concept':25s} {'Split':8s} ", end="")
            for cls in [0, 1, 2]:
                print(f"Class {cls}:", end="")
            print()
            print("-" * 60)
            
            for i, concept in enumerate(concept_columns):
                # Train distribution
                train_unique, train_counts = np.unique(class_indices[train_idx, i], return_counts=True)
                train_dict = {cls: 0 for cls in [0, 1, 2]}
                for cls, cnt in zip(train_unique, train_counts):
                    train_dict[cls] = cnt
                print(f"{concept:25s} {'Train':8s} ", end="")
                for cls in [0, 1, 2]:
                    print(f"{train_dict[cls]:8d}  ", end="")
                print()
                
                # Validation distribution
                val_unique, val_counts = np.unique(class_indices[val_idx, i], return_counts=True)
                val_dict = {cls: 0 for cls in [0, 1, 2]}
                for cls, cnt in zip(val_unique, val_counts):
                    val_dict[cls] = cnt
                print(f"{'':25s} {'Val':8s} ", end="")
                for cls in [0, 1, 2]:
                    print(f"{val_dict[cls]:8d}  ", end="")
                print()
            
            print("=" * 60)
            return train_idx, val_idx
        
        # If validation failed, print warning and retry
        if attempt < 5:  # Only print first few attempts
            print(f"⚠️  Attempt {attempt + 1} failed: Missing classes detected")
            for missing in missing_classes:
                if missing['train_missing']:
                    print(f"   {missing['concept']}: Missing in train: {missing['train_missing']}")
                if missing['val_missing']:
                    print(f"   {missing['concept']}: Missing in val: {missing['val_missing']}")
    
    # If we exhausted all attempts, raise an error
    raise ValueError(
        f"Could not create a valid split after {max_attempts} attempts. "
        f"Some concepts may not have enough samples for all classes. "
        f"Last missing classes: {missing_classes}"
    )


def train_concept_predictor():
    """Train the concept predictor using pretrained encoder."""
    print("Starting concept prediction training...")
    
    # ------------------------------------------------------------------
    # 1. Load raw submission data (sensor windows + concept labels)
    # ------------------------------------------------------------------
    windows, concept_labels, concept_columns = load_and_prepare_data()

    # ------------------------------------------------------------------
    # 2. Standardize each axis using the pretraining scaler (or create one)
    # ------------------------------------------------------------------
    scaler_path = CURRENT_DIR / 'improved_scaler.pkl'

    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Loaded existing scaler from pretraining")
    except:
        print("Creating new scaler...")
        # Flatten all windows to create scaler
        all_sensor_data = np.vstack(windows)
        scaler = StandardScaler()
        scaler.fit(all_sensor_data)
        # Save the new scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print("New scaler created and saved")
    
    # Apply scaling window by window to preserve structure
    windows_scaled = [scaler.transform(window) for window in windows]

    # ------------------------------------------------------------------
    # 3. Create a stratified train/validation split (all concepts covered)
    # ------------------------------------------------------------------
    train_idx, val_idx = create_stratified_split(windows_scaled, concept_labels, concept_columns, test_size=0.2)

    train_windows = [windows_scaled[i] for i in train_idx]
    train_labels = concept_labels[train_idx]
    val_windows = [windows_scaled[i] for i in val_idx]
    val_labels = concept_labels[val_idx]

    # ------------------------------------------------------------------
    # 4. Optional data augmentation (jitter, scaling, rotation + rare-class boost)
    # ------------------------------------------------------------------
    # Results: ~0.8447 AUROC with augmentation vs ~0.8089 without (significant improvement)
    USE_AUGMENTATION = True  # Set to False to disable augmentation

    if USE_AUGMENTATION:
        print("\n" + "=" * 60)
        print("PREPARING TRAINING DATA WITH AUGMENTATION")
        print("=" * 60)
        train_windows_aug, train_labels_aug = augment_training_data(
            train_windows,
            train_labels,
            concept_columns,
            base_factor=10,  # 10x base augmentation
            rare_class_boost=5  # 5x additional augmentation for rare classes
        )
    else:
        print("\n" + "=" * 60)
        print("PREPARING TRAINING DATA (NO AUGMENTATION)")
        print("=" * 60)
        train_windows_aug = train_windows
        train_labels_aug = train_labels
        print(f"Using original training samples: {len(train_windows_aug)}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 5. Wrap numpy data in PyTorch datasets and deterministic loaders
    # ------------------------------------------------------------------
    train_dataset = ConceptDataset(train_windows_aug, train_labels_aug)
    val_dataset = ConceptDataset(val_windows, val_labels)  # No augmentation for validation

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # ------------------------------------------------------------------
    # 6. Build the concept predictor (pretrained encoder + classification heads)
    # ------------------------------------------------------------------
    pretrained_encoder_path = CURRENT_DIR / 'improved_pretrained_encoder.pth'
    model = PretrainedConceptPredictor(
        pretrained_encoder_path=pretrained_encoder_path,
        input_dim=3,
        hidden_dim=64,
        num_concepts=5,
        num_classes=3,  # 3 classes: 0.0, 0.5, 1.0
        freeze_encoder=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # ------------------------------------------------------------------
    # 7. Train with early stopping, tracking per-epoch metrics for all concepts
    # ------------------------------------------------------------------
    num_epochs = 50
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training loop ---------------------------------------------------
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)  # List of 5 tensors, each (batch_size, 3)

            # Multi-task loss: average over the five concept heads
            loss = 0.0
            for i in range(5):
                loss += criterion(outputs[i], batch_y[:, i])
            loss = loss / 5.0

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation loop -------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_predictions = []  # List of lists: [[pred_concept0], [pred_concept1], ...]
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)

                loss = 0.0
                for i in range(5):
                    loss += criterion(outputs[i], batch_y[:, i])
                loss = loss / 5.0
                val_loss += loss.item()
                
                # Store predictions per concept for later metric computation
                batch_predictions = [torch.argmax(outputs[i], dim=1).cpu().numpy() for i in range(5)]
                if len(val_predictions) == 0:
                    val_predictions = [[pred] for pred in batch_predictions]
                else:
                    for i in range(5):
                        val_predictions[i].append(batch_predictions[i])

                val_targets.append(batch_y.cpu().numpy())
        
        # Aggregate metrics -----------------------------------------------
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        val_predictions = [np.concatenate(preds) for preds in val_predictions]
        val_targets = np.concatenate(val_targets)

        accuracies = []
        for i in range(5):
            acc = accuracy_score(val_targets[:, i], val_predictions[i])
            accuracies.append(acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Accuracies: {[f"{acc:.3f}" for acc in accuracies]}')
        print(f'  Concepts: {concept_columns}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = CURRENT_DIR / 'best_concept_predictor.pth'
            torch.save(model.state_dict(), model_path)
            print(f'  New best model saved to {model_path}!')
        
        scheduler.step(val_loss)
        
        if epoch > 10 and val_loss > best_val_loss * 1.1:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # ------------------------------------------------------------------
    # 8. Reload best checkpoint and compute final evaluation artefacts
    # ------------------------------------------------------------------
    model_path = CURRENT_DIR / 'best_concept_predictor.pth'
    model.load_state_dict(torch.load(model_path))

    model.eval()
    final_predictions = []  # List of lists: [[pred_concept0], [pred_concept1], ...]
    final_pred_probs = []   # List of lists: [[prob_concept0], [prob_concept1], ...] for AUROC
    final_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            # Get prediction probabilities (for AUROC)
            batch_pred_probs = [torch.softmax(outputs[i], dim=1).cpu().numpy() for i in range(5)]
            
            # Get predictions (class indices)
            batch_predictions = [np.argmax(prob, axis=1) for prob in batch_pred_probs]
            
            if len(final_predictions) == 0:
                final_predictions = [[pred] for pred in batch_predictions]
                final_pred_probs = [[prob] for prob in batch_pred_probs]
            else:
                for i in range(5):
                    final_predictions[i].append(batch_predictions[i])
                    final_pred_probs[i].append(batch_pred_probs[i])
            
            final_targets.append(batch_y.cpu().numpy())
    
    # Concatenate predictions and targets
    final_predictions = [np.concatenate(preds) for preds in final_predictions]
    final_pred_probs = [np.concatenate(probs, axis=0) for probs in final_pred_probs]
    final_targets = np.concatenate(final_targets)
    
    # Calculate final metrics
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    
    auroc_scores = []
    
    for i, concept in enumerate(concept_columns):
        y_true = final_targets[:, i]
        y_pred = final_predictions[i]
        y_probs = final_pred_probs[i]
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Calculate AUROC - handle missing classes
        # Use one-vs-rest approach: calculate AUROC for each class separately
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        
        if n_classes < 2:
            # Not enough classes for AUROC
            auroc = 0.5
            print(f"⚠️  {concept}: Only {n_classes} class(es) present, cannot calculate AUROC")
        else:
            # Calculate one-vs-rest AUROC for each class that's present
            auroc_per_class = []
            expected_classes = np.arange(y_probs.shape[1])  # [0, 1, 2] for 3-class problem
            
            for class_idx in expected_classes:
                if class_idx in unique_classes:
                    # One-vs-rest: this class vs all others
                    y_binary = (y_true == class_idx).astype(int)
                    # Use probability of this class
                    try:
                        class_auroc = roc_auc_score(y_binary, y_probs[:, class_idx])
                        auroc_per_class.append(class_auroc)
                    except Exception as e:
                        # Skip if calculation fails (e.g., only one class in binary split)
                        pass
            
            if auroc_per_class:
                # Average AUROC across all classes (macro average)
                auroc = np.mean(auroc_per_class)
            else:
                # Fallback if no valid AUROC could be calculated
                auroc = 0.5
                print(f"⚠️  {concept}: Could not calculate AUROC for any class")
        
        auroc_scores.append(auroc)
        
        # Print class distribution for debugging
        class_counts = {int(c): int(np.sum(y_true == c)) for c in unique_classes}
        
        print(f"\n{concept.upper()}:")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"  F1-Score (weighted): {f1:.4f}")
        print(f"  AUROC:    {auroc:.4f}")
        print(f"  Classes present: {sorted(unique_classes)} (counts: {class_counts})")
        
        # Print confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"  Confusion Matrix:")
        print(f"    {cm}")
    
    # Overall metrics
    overall_acc = np.mean([accuracy_score(final_targets[:, i], final_predictions[i]) for i in range(5)])
    overall_auroc = np.mean(auroc_scores)
    
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE")
    print("=" * 60)
    print(f"Average Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"Average AUROC:    {overall_auroc:.4f}")
    print("=" * 60)
    
    # Highlight rule-based concepts
    print("\n📊 RULE-BASED CONCEPTS PERFORMANCE:")
    print("-" * 60)
    mv_idx = concept_columns.index('movement_variability')
    mc_idx = concept_columns.index('movement_consistency')
    
    print(f"Movement Variability:")
    print(f"  Accuracy: {accuracy_score(final_targets[:, mv_idx], final_predictions[mv_idx]):.4f}")
    print(f"  AUROC:    {auroc_scores[mv_idx]:.4f}")
    
    print(f"\nMovement Consistency:")
    print(f"  Accuracy: {accuracy_score(final_targets[:, mc_idx], final_predictions[mc_idx]):.4f}")
    print(f"  AUROC:    {auroc_scores[mc_idx]:.4f}")
    print("=" * 60)
    
    # Plot training curves
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training/validation loss
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy and AUROC comparison
    accuracies = [accuracy_score(final_targets[:, i], final_predictions[i]) for i in range(5)]
    x = np.arange(len(concept_columns))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7, color='steelblue')
    bars2 = ax2.bar(x + width/2, auroc_scores, width, label='AUROC', alpha=0.7, color='coral')
    
    ax2.set_xlabel('Concepts')
    ax2.set_ylabel('Score')
    ax2.set_title('Accuracy and AUROC by Concept')
    ax2.set_xticks(x)
    ax2.set_xticklabels(concept_columns, rotation=45, ha='right')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    results_path = CURRENT_DIR / 'concept_prediction_results.png'
    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nModel saved as '{model_path}'")
    print(f"Results plot saved as '{results_path}'")


if __name__ == "__main__":
    train_concept_predictor()
