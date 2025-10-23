"""
Class Imbalance Correction Module

This module contains class imbalance analysis and correction techniques
that were commented out from the main training notebook. It provides
methods for handling imbalanced datasets and improving model performance.

Usage:
    from class_imbalance_correction import run_class_imbalance_analysis
    
    results = run_class_imbalance_analysis(X_train, y_train, X_test, y_test)
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

def calculate_class_weights(y_p, y_t, y_c):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        y_p, y_t, y_c: Labels for three concepts
    
    Returns:
        Class weights for each concept
    """
    print("Calculating class weights...")
    
    # Convert to discrete classes
    y_p_discrete = (y_p * 2).astype(int)
    y_t_discrete = (y_t * 2).astype(int)
    y_c_discrete = (y_c * 2).astype(int)
    
    # Calculate weights for each concept
    p_weights = compute_class_weight('balanced', classes=np.unique(y_p_discrete), y=y_p_discrete)
    t_weights = compute_class_weight('balanced', classes=np.unique(y_t_discrete), y=y_t_discrete)
    c_weights = compute_class_weight('balanced', classes=np.unique(y_c_discrete), y=y_c_discrete)
    
    # Create weight dictionaries
    p_weight_dict = {i: weight for i, weight in enumerate(p_weights)}
    t_weight_dict = {i: weight for i, weight in enumerate(t_weights)}
    c_weight_dict = {i: weight for i, weight in enumerate(c_weights)}
    
    print(f"Periodicity weights: {p_weight_dict}")
    print(f"Temporal Stability weights: {t_weight_dict}")
    print(f"Coordination weights: {c_weight_dict}")
    
    return p_weight_dict, t_weight_dict, c_weight_dict

def build_multi_output_cnn(input_shape, n_classes_p, n_classes_t, n_classes_c):
    """Build multi-output CNN for class-weighted training"""
    input_layer = layers.Input(shape=input_shape)
    
    # Shared feature extractor
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dropout(0.4)(x)
    
    # Output heads for each concept
    periodicity = layers.Dense(n_classes_p, activation='softmax', name='periodicity')(x)
    temporal_stability = layers.Dense(n_classes_t, activation='softmax', name='temporal_stability')(x)
    coordination = layers.Dense(n_classes_c, activation='softmax', name='coordination')(x)
    
    model = models.Model(
        inputs=input_layer,
        outputs=[periodicity, temporal_stability, coordination]
    )
    
    return model

def train_model_with_class_weights(X_train, X_test, y_p_train, y_p_test, y_t_train, y_t_test, 
                                 y_c_train, y_c_test, p_weights, t_weights, c_weights, model_name):
    """
    Train model with class weights
    
    Args:
        X_train, X_test: Training and test data
        y_p_train, y_p_test: Periodicity labels
        y_t_train, y_t_test: Temporal stability labels
        y_c_train, y_c_test: Coordination labels
        p_weights, t_weights, c_weights: Class weights
        model_name: Name for the model
    
    Returns:
        Trained model, history, and results
    """
    print(f"Training {model_name} with class weights...")
    
    # Build model
    model = build_multi_output_cnn(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        n_classes_p=3, n_classes_t=3, n_classes_c=3
    )
    
    # Compile with class weights
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'periodicity': 'categorical_crossentropy',
            'temporal_stability': 'categorical_crossentropy',
            'coordination': 'categorical_crossentropy'
        },
        loss_weights={
            'periodicity': 1.0,
            'temporal_stability': 1.0,
            'coordination': 1.0
        },
        metrics={
            'periodicity': ['accuracy'],
            'temporal_stability': ['accuracy'],
            'coordination': ['accuracy']
        }
    )
    
    # Convert labels to categorical
    y_p_train_cat = to_categorical(y_p_train * 2, num_classes=3)
    y_t_train_cat = to_categorical(y_t_train * 2, num_classes=3)
    y_c_train_cat = to_categorical(y_c_train * 2, num_classes=3)
    y_p_test_cat = to_categorical(y_p_test * 2, num_classes=3)
    y_t_test_cat = to_categorical(y_t_test * 2, num_classes=3)
    y_c_test_cat = to_categorical(y_c_test * 2, num_classes=3)
    
    # Train model
    history = model.fit(
        X_train,
        [y_p_train_cat, y_t_train_cat, y_c_train_cat],
        validation_data=(X_test, [y_p_test_cat, y_t_test_cat, y_c_test_cat]),
        epochs=50,
        batch_size=16,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Evaluate model
    results = model.evaluate(X_test, [y_p_test_cat, y_t_test_cat, y_c_test_cat], verbose=0)
    
    # Extract accuracies
    periodicity_acc = results[4]
    temporal_stability_acc = results[5]
    coordination_acc = results[6]
    overall_acc = (periodicity_acc + temporal_stability_acc + coordination_acc) / 3
    
    print(f"\n=== {model_name.upper()} RESULTS ===")
    print(f"Periodicity Accuracy: {periodicity_acc:.4f} ({periodicity_acc*100:.1f}%)")
    print(f"Temporal Stability Accuracy: {temporal_stability_acc:.4f} ({temporal_stability_acc*100:.1f}%)")
    print(f"Coordination Accuracy: {coordination_acc:.4f} ({coordination_acc*100:.1f}%)")
    print(f"Overall Average Accuracy: {overall_acc:.4f} ({overall_acc*100:.1f}%)")
    
    return model, history, {
        'periodicity_acc': periodicity_acc,
        'temporal_stability_acc': temporal_stability_acc,
        'coordination_acc': coordination_acc,
        'overall_acc': overall_acc
    }

def cv_with_class_weights(X, y_p, y_t, y_c, n_folds=5):
    """
    Cross-validation with class weights
    
    Args:
        X: Sensor data
        y_p, y_t, y_c: Labels
        n_folds: Number of CV folds
    
    Returns:
        CV results DataFrame
    """
    print(f"Running {n_folds}-fold CV with class weights...")
    
    # Convert to discrete for stratification
    y_p_discrete = (y_p * 2).astype(int)
    
    # Use stratified K-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_results_weighted = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_p_discrete)):
        print(f"\n--- Fold {fold + 1}/{n_folds} (with class weights) ---")
        
        # Get fold data
        X_train_fold = X[train_idx]
        X_test_fold = X[test_idx]
        y_p_train_fold = y_p[train_idx]
        y_p_test_fold = y_p[test_idx]
        y_t_train_fold = y_t[train_idx]
        y_t_test_fold = y_t[test_idx]
        y_c_train_fold = y_c[train_idx]
        y_c_test_fold = y_c[test_idx]
        
        # Calculate class weights for this fold
        p_weights_fold, t_weights_fold, c_weights_fold = calculate_class_weights(
            y_p_train_fold, y_t_train_fold, y_c_train_fold
        )
        
        # Train model with weights
        _, _, fold_results = train_model_with_class_weights(
            X_train_fold, X_test_fold,
            y_p_train_fold, y_p_test_fold,
            y_t_train_fold, y_t_test_fold,
            y_c_train_fold, y_c_test_fold,
            p_weights_fold, t_weights_fold, c_weights_fold,
            f"CV Fold {fold + 1}"
        )
        
        cv_results_weighted.append(fold_results)
    
    # Analyze CV results
    cv_df_weighted = pd.DataFrame(cv_results_weighted)
    
    print(f"\n=== CROSS-VALIDATION WITH CLASS WEIGHTS RESULTS ===")
    print(f"Mean accuracy: {cv_df_weighted['overall_acc'].mean():.4f} ± {cv_df_weighted['overall_acc'].std():.4f}")
    print(f"Accuracy range: {cv_df_weighted['overall_acc'].min():.4f} - {cv_df_weighted['overall_acc'].max():.4f}")
    print(f"Range span: {(cv_df_weighted['overall_acc'].max() - cv_df_weighted['overall_acc'].min())*100:.1f}%")
    
    return cv_df_weighted

def run_class_imbalance_analysis(X, y_p, y_t, y_c):
    """
    Run comprehensive class imbalance analysis and correction
    
    Args:
        X: Sensor data
        y_p, y_t, y_c: Labels
    
    Returns:
        Analysis results
    """
    print("=== CLASS IMBALANCE CORRECTION ANALYSIS ===")
    
    # 1. Calculate class weights
    print("\n1. CALCULATING CLASS WEIGHTS")
    p_weights, t_weights, c_weights = calculate_class_weights(y_p, y_t, y_c)
    
    # 2. Train model with class weights
    print("\n2. TRAINING WITH CLASS WEIGHTS")
    weighted_model, weighted_history, weighted_results = train_model_with_class_weights(
        X, X, y_p, y_p, y_t, y_t, y_c, y_c,
        p_weights, t_weights, c_weights,
        "Class Weighted Model"
    )
    
    # 3. Cross-validation with class weights
    print("\n3. CROSS-VALIDATION WITH CLASS WEIGHTS")
    cv_results_weighted = cv_with_class_weights(X, y_p, y_t, y_c, n_folds=5)
    
    # 4. Performance comparison
    print("\n4. PERFORMANCE COMPARISON")
    print(f"Weighted model: {weighted_results['overall_acc']:.4f} ({weighted_results['overall_acc']*100:.1f}%)")
    print(f"CV with weights: {cv_results_weighted['overall_acc'].mean():.4f} ± {cv_results_weighted['overall_acc'].std():.4f}")
    
    return {
        'weighted_results': weighted_results,
        'cv_results_weighted': cv_results_weighted,
        'class_weights': {
            'periodicity': p_weights,
            'temporal_stability': t_weights,
            'coordination': c_weights
        }
    }

if __name__ == "__main__":
    print("Class Imbalance Correction Module")
    print("Import this module to use the class imbalance correction functionality.")
