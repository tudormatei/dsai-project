"""
Hyperparameter Grid Search Module

This module contains the hyperparameter grid search code that was commented out
from the main training notebook. It provides comprehensive hyperparameter tuning
for the CNN model architecture.

Usage:
    from hyperparameter_search import run_hyperparameter_search
    
    results = run_hyperparameter_search(X_train, y_train, X_test, y_test)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tensorflow import keras
from tensorflow.keras import layers, models
import time

def build_cnn_for_search(input_shape, conv_filters_1, conv_filters_2, dropout_rate, n_classes):
    """Build CNN model for hyperparameter search"""
    input_layer = layers.Input(shape=input_shape)
    
    # Conv layer 1
    x = layers.Conv1D(conv_filters_1, 3, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Conv layer 2
    x = layers.Conv1D(conv_filters_2, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dropout
    x = layers.Dropout(dropout_rate)(x)
    
    # Output head
    output = layers.Dense(n_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=output)
    return model

def run_hyperparameter_search(X_train, y_train, X_test, y_test, max_combinations=50):
    """
    Run hyperparameter grid search for CNN model
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        max_combinations: Maximum number of combinations to test
    
    Returns:
        DataFrame with results
    """
    print("=== HYPERPARAMETER GRID SEARCH ===")
    
    # Define parameter grid
    param_grid = {
        'conv_filters_1': [8, 16, 32],
        'conv_filters_2': [16, 32, 64],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [16, 32, 64]
    }
    
    # Create parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    
    # Limit combinations if too many
    if len(param_combinations) > max_combinations:
        np.random.seed(42)
        param_combinations = np.random.choice(
            param_combinations, max_combinations, replace=False
        ).tolist()
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\n--- Combination {i+1}/{len(param_combinations)} ---")
        print(f"Parameters: {params}")
        
        try:
            # Build model
            model = build_cnn_for_search(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                conv_filters_1=params['conv_filters_1'],
                conv_filters_2=params['conv_filters_2'],
                dropout_rate=params['dropout_rate'],
                n_classes=3
            )
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=params['batch_size'],
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ]
            )
            
            training_time = time.time() - start_time
            
            # Evaluate model
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            
            # Store results
            result = {
                'conv_filters_1': params['conv_filters_1'],
                'conv_filters_2': params['conv_filters_2'],
                'dropout_rate': params['dropout_rate'],
                'learning_rate': params['learning_rate'],
                'batch_size': params['batch_size'],
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'training_time': training_time,
                'epochs_trained': len(history.history['loss']),
                'model_params': model.count_params()
            }
            
            results.append(result)
            print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
            print(f"Training Time: {training_time:.1f}s")
            
        except Exception as e:
            print(f"Error with combination {i+1}: {str(e)}")
            continue
    
    # Convert to DataFrame and sort by accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_accuracy', ascending=False)
    
    print(f"\n=== GRID SEARCH RESULTS ===")
    print(f"Best accuracy: {results_df.iloc[0]['test_accuracy']:.4f} ({results_df.iloc[0]['test_accuracy']*100:.1f}%)")
    print(f"Best parameters:")
    best_params = results_df.iloc[0]
    for param in ['conv_filters_1', 'conv_filters_2', 'dropout_rate', 'learning_rate', 'batch_size']:
        print(f"  {param}: {best_params[param]}")
    
    return results_df

if __name__ == "__main__":
    print("Hyperparameter Search Module")
    print("Import this module to use the hyperparameter search functionality.")
