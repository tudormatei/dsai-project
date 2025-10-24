"""
Integration code for quick_train_cnn.ipynb
Add this as a new cell in your notebook
"""

# =============================================================================
# NEW CELL: Integrate Fine-Tuned Concept Predictor
# =============================================================================

import torch
import numpy as np
import pandas as pd
import pickle
import sys
import os

# Add the pretraining directory to path
sys.path.append('/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining')

# Import the concept predictor
from fine_tune_concepts import PretrainedConceptPredictor

class ConceptPredictorWrapper:
    """Wrapper for the fine-tuned concept prediction model."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.device = torch.device('cpu')
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned concept prediction model."""
        print("Loading fine-tuned concept predictor...")
        
        # Initialize model
        self.model = PretrainedConceptPredictor(
            pretrained_encoder_path='/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining/improved_pretrained_encoder.pth',
            input_dim=3,
            hidden_dim=64,
            num_concepts=5,
            freeze_encoder=True
        )
        
        # Load fine-tuned weights
        model_path = '/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining/best_concept_predictor.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Load scaler
        scaler_path = '/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining/improved_scaler.pkl'
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print("✓ Concept predictor loaded successfully!")
    
    def predict_concepts_from_windows(self, X_windows):
        """
        Predict concepts from pre-processed windows.
        
        Args:
            X_windows: Pre-processed windows (n_windows, window_size, 3)
            
        Returns:
            concept_predictions: Array of concept predictions (n_windows, 5)
        """
        # Convert to tensor and transpose for Conv1d
        windows_tensor = torch.FloatTensor(X_windows.transpose(0, 2, 1)).to(self.device)
        
        # Predict concepts
        with torch.no_grad():
            predictions = self.model(windows_tensor)
            predictions = predictions.cpu().numpy()
        
        return predictions
    
    def get_concept_features(self, X_windows):
        """
        Get concept predictions as additional features for your CNN.
        
        Args:
            X_windows: Pre-processed windows (n_windows, window_size, 3)
            
        Returns:
            concept_features: Concept predictions as features (n_windows, 5)
        """
        return self.predict_concepts_from_windows(X_windows)

# Initialize the concept predictor
print("Initializing concept predictor...")
concept_predictor = ConceptPredictorWrapper()

# =============================================================================
# NEW CELL: Enhanced CNN with Concept Features
# =============================================================================

def build_enhanced_cnn_with_concepts(input_shape, n_classes_p, n_classes_t, n_classes_c, contextual_config):
    """
    Enhanced CNN that uses both raw sensor data and concept predictions.
    
    This model combines:
    1. Traditional CNN processing of sensor data
    2. Concept predictions from the fine-tuned model
    3. Contextual features
    """
    # Input layer for sensor data
    sensor_input = layers.Input(shape=input_shape, name='sensor_input')
    
    # Input layer for concept predictions
    concept_input = layers.Input(shape=(5,), name='concept_input')
    
    # CNN processing of sensor data
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(sensor_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Combine sensor features with concept features
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Concatenate with concept predictions
    combined_features = layers.Concatenate()([x, concept_input])
    combined_features = layers.Dense(64, activation='relu')(combined_features)
    combined_features = layers.Dropout(0.3)(combined_features)
    
    # Static posture context
    static_context = layers.Dense(1, activation='sigmoid', name='static_context')(combined_features)
    
    # Output heads for each concept
    outputs = []
    output_names = []
    
    # Periodicity (3 classes)
    if 'periodicity' in contextual_config and contextual_config['periodicity']:
        periodicity_output = layers.Dense(n_classes_p, activation='softmax', name='periodicity')(combined_features)
        outputs.append(periodicity_output)
        output_names.append('periodicity')
    
    # Temporal stability (3 classes)
    if 'temporal_stability' in contextual_config and contextual_config['temporal_stability']:
        temporal_output = layers.Dense(n_classes_t, activation='softmax', name='temporal_stability')(combined_features)
        outputs.append(temporal_output)
        output_names.append('temporal_stability')
    
    # Coordination (3 classes)
    if 'coordination' in contextual_config and contextual_config['coordination']:
        coordination_output = layers.Dense(n_classes_c, activation='softmax', name='coordination')(combined_features)
        outputs.append(coordination_output)
        output_names.append('coordination')
    
    # Motion intensity (regression)
    if 'motion_intensity' in contextual_config and contextual_config['motion_intensity']:
        motion_intensity_output = layers.Dense(1, activation='sigmoid', name='motion_intensity')(combined_features)
        outputs.append(motion_intensity_output)
        output_names.append('motion_intensity')
    
    # Vertical dominance (regression)
    if 'vertical_dominance' in contextual_config and contextual_config['vertical_dominance']:
        vertical_dominance_output = layers.Dense(1, activation='sigmoid', name='vertical_dominance')(combined_features)
        outputs.append(vertical_dominance_output)
        output_names.append('vertical_dominance')
    
    # Add static context to outputs
    outputs.append(static_context)
    output_names.append('static_context')
    
    # Create model
    model = models.Model(inputs=[sensor_input, concept_input], outputs=outputs)
    
    print(f"Enhanced model architecture:")
    print(f"  Input 1: Sensor data {input_shape}")
    print(f"  Input 2: Concept predictions (5,)")
    print(f"  Outputs: {output_names}")
    
    return model

# =============================================================================
# NEW CELL: Training with Concept Features
# =============================================================================

def train_with_concept_features():
    """
    Train the enhanced CNN using concept predictions as additional features.
    """
    print("Training Enhanced CNN with Concept Features")
    print("=" * 60)
    
    # Load your existing data (assuming you have X_train, X_test, etc.)
    # This is where you'd load your preprocessed data from the notebook
    
    # Get concept predictions for training data
    print("Generating concept predictions for training data...")
    X_train_concepts = concept_predictor.get_concept_features(X_train_aug)
    print(f"Training concept features shape: {X_train_concepts.shape}")
    
    # Get concept predictions for test data
    print("Generating concept predictions for test data...")
    X_test_concepts = concept_predictor.get_concept_features(X_test)
    print(f"Test concept features shape: {X_test_concepts.shape}")
    
    # Build enhanced model
    print("Building enhanced model...")
    enhanced_model = build_enhanced_cnn_with_concepts(
        input_shape=(X_train_aug.shape[1], X_train_aug.shape[2]),
        n_classes_p=3, n_classes_t=3, n_classes_c=3,
        contextual_config=contextual_config
    )
    
    # Compile model
    enhanced_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'periodicity': 'categorical_crossentropy',
            'temporal_stability': 'categorical_crossentropy', 
            'coordination': 'categorical_crossentropy',
            'motion_intensity': 'mse',
            'vertical_dominance': 'mse',
            'static_context': 'binary_crossentropy'
        },
        loss_weights={
            'periodicity': 1.0,
            'temporal_stability': 1.0,
            'coordination': 1.0,
            'motion_intensity': 1.0,
            'vertical_dominance': 1.0,
            'static_context': 0.5
        },
        metrics={
            'periodicity': 'accuracy',
            'temporal_stability': 'accuracy',
            'coordination': 'accuracy',
            'motion_intensity': 'mae',
            'vertical_dominance': 'mae',
            'static_context': 'accuracy'
        }
    )
    
    # Prepare training data with concept features
    train_inputs = [X_train_aug, X_train_concepts]
    train_targets = {
        'periodicity': y_p_train_cat,
        'temporal_stability': y_t_train_cat,
        'coordination': y_c_train_cat,
        'motion_intensity': y_mi_train,
        'vertical_dominance': y_vd_train,
        'static_context': y_sp_context_train_cat
    }
    
    # Prepare validation data
    val_inputs = [X_test, X_test_concepts]
    val_targets = {
        'periodicity': y_p_test_cat,
        'temporal_stability': y_t_test_cat,
        'coordination': y_c_test_cat,
        'motion_intensity': y_mi_test,
        'vertical_dominance': y_vd_test,
        'static_context': y_sp_context_test_cat
    }
    
    # Train the enhanced model
    print("Training enhanced model...")
    history = enhanced_model.fit(
        train_inputs, train_targets,
        validation_data=(val_inputs, val_targets),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )
    
    # Evaluate the model
    print("Evaluating enhanced model...")
    results = enhanced_model.evaluate(val_inputs, val_targets, verbose=0)
    
    print("Enhanced Model Results:")
    print("=" * 40)
    for i, metric in enumerate(enhanced_model.metrics_names):
        print(f"{metric}: {results[i]:.4f}")
    
    return enhanced_model, history

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

print("""
HOW TO USE THIS IN YOUR NOTEBOOK:

1. Add the ConceptPredictorWrapper cell to your notebook
2. Add the build_enhanced_cnn_with_concepts function
3. Add the train_with_concept_features function
4. Run the training function

The enhanced model will:
- Use your existing sensor data preprocessing
- Generate concept predictions using the fine-tuned model
- Combine sensor features with concept predictions
- Train a more powerful model that leverages both raw data and learned concepts

This approach gives you:
✓ Better feature representation from pretrained concepts
✓ Transfer learning benefits
✓ More robust predictions
✓ Interpretable concept-based features
""")
