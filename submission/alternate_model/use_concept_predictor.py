"""
How to use the fine-tuned concept prediction model
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

class ConceptPredictor:
    """Wrapper class for using the fine-tuned concept prediction model."""
    
    def __init__(self, model_path, scaler_path, device='cpu'):
        """
        Initialize the concept predictor.
        
        Args:
            model_path: Path to the fine-tuned model weights
            scaler_path: Path to the scaler used for preprocessing
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)
        
    def _load_model(self, model_path):
        """Load the fine-tuned concept prediction model."""
        # Import the model architecture
        import sys
        sys.path.append('/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining')
        from fine_tune_concepts import PretrainedConceptPredictor
        
        # Initialize model
        model = PretrainedConceptPredictor(
            pretrained_encoder_path='/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining/improved_pretrained_encoder.pth',
            input_dim=3,
            hidden_dim=64,
            num_concepts=5,
            freeze_encoder=True
        )
        
        # Load fine-tuned weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model = model.to(self.device)
        
        return model
    
    def _load_scaler(self, scaler_path):
        """Load the scaler used for preprocessing."""
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    
    def predict_concepts(self, sensor_data, window_size=128, stride=32):
        """
        Predict concepts from sensor data.
        
        Args:
            sensor_data: Raw sensor data (n_samples, 3) - x, y, z axes
            window_size: Size of time windows for prediction
            stride: Stride for window creation
            
        Returns:
            predictions: Array of concept predictions (n_windows, 5)
            window_info: Information about each window
        """
        # Scale the data
        sensor_data_scaled = self.scaler.transform(sensor_data)
        
        # Create windows
        windows = []
        window_starts = []
        
        for i in range(0, len(sensor_data_scaled) - window_size + 1, stride):
            window = sensor_data_scaled[i:i + window_size]
            if len(window) == window_size:
                windows.append(window.T)  # Transpose for Conv1d
                window_starts.append(i)
        
        if not windows:
            return np.array([]), []
        
        # Convert to tensor
        windows_tensor = torch.FloatTensor(np.array(windows)).to(self.device)
        
        # Predict concepts
        with torch.no_grad():
            predictions = self.model(windows_tensor)
            predictions = predictions.cpu().numpy()
        
        # Create window info
        window_info = []
        for i, start in enumerate(window_starts):
            window_info.append({
                'window_idx': i,
                'start_sample': start,
                'end_sample': start + window_size,
                'start_time': start / 50.0,  # Assuming 50Hz sampling rate
                'end_time': (start + window_size) / 50.0
            })
        
        return predictions, window_info
    
    def predict_single_window(self, window_data):
        """
        Predict concepts for a single window.
        
        Args:
            window_data: Single window of sensor data (window_size, 3)
            
        Returns:
            concepts: Dictionary with concept predictions
        """
        # Scale the data
        window_scaled = self.scaler.transform(window_data)
        
        # Convert to tensor
        window_tensor = torch.FloatTensor(window_scaled.T).unsqueeze(0).to(self.device)
        
        # Predict concepts
        with torch.no_grad():
            predictions = self.model(window_tensor)
            predictions = predictions.cpu().numpy()[0]
        
        # Create concept dictionary
        concept_names = ['periodicity', 'temporal_stability', 'coordination', 
                        'motion_intensity', 'vertical_dominance']
        
        concepts = {}
        for i, name in enumerate(concept_names):
            concepts[name] = float(predictions[i])
        
        return concepts


def example_usage():
    """Example of how to use the concept predictor."""
    
    # Initialize the predictor
    predictor = ConceptPredictor(
        model_path='/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining/best_concept_predictor.pth',
        scaler_path='/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining/improved_scaler.pkl',
        device='cpu'
    )
    
    # Example 1: Load and predict from CSV file
    print("Example 1: Predicting from CSV file")
    print("=" * 50)
    
    # Load sensor data
    raw_data = pd.read_csv('/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/rule_based_labeling/raw_with_features.csv')
    sensor_data = raw_data[['x-axis', 'y-axis', 'z-axis']].values
    
    print(f"Loaded sensor data: {sensor_data.shape}")
    
    # Predict concepts
    predictions, window_info = predictor.predict_concepts(sensor_data, window_size=128, stride=64)
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Concept predictions shape: {predictions.shape}")
    
    # Show first few predictions
    concept_names = ['periodicity', 'temporal_stability', 'coordination', 
                    'motion_intensity', 'vertical_dominance']
    
    print("\nFirst 5 predictions:")
    for i in range(min(5, len(predictions))):
        print(f"Window {i}:")
        for j, concept in enumerate(concept_names):
            print(f"  {concept}: {predictions[i][j]:.3f}")
        print()
    
    # Example 2: Predict from single window
    print("Example 2: Single window prediction")
    print("=" * 50)
    
    # Take a single window
    window_data = sensor_data[:128]  # First 128 samples
    concepts = predictor.predict_single_window(window_data)
    
    print("Single window concept predictions:")
    for concept, value in concepts.items():
        print(f"  {concept}: {value:.3f}")
    
    return predictions, window_info, concepts


if __name__ == "__main__":
    # Run example
    predictions, window_info, single_concepts = example_usage()
