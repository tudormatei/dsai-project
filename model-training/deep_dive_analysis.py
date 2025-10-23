"""
Deep Dive Analysis Module

This module contains comprehensive analysis code for model performance,
cross-validation, and advanced evaluation techniques that were commented out
from the main training notebook.

Usage:
    from deep_dive_analysis import run_deep_dive_analysis
    
    results = run_deep_dive_analysis(X, y_p, y_t, y_c)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns

def filter_outliers_by_activity(X, y_p, y_t, y_c, activity_col=None, outlier_threshold=2.0):
    """
    Filter outliers based on activity patterns
    
    Args:
        X: Sensor data
        y_p, y_t, y_c: Labels
        activity_col: Activity column (if available)
        outlier_threshold: Z-score threshold for outlier detection
    
    Returns:
        Filtered data without outliers
    """
    print("Filtering outliers by activity patterns...")
    
    # Calculate activity intensity (magnitude of acceleration)
    activity_intensity = np.sqrt(np.sum(X**2, axis=(1, 2)))
    
    # Calculate z-scores
    z_scores = np.abs((activity_intensity - np.mean(activity_intensity)) / np.std(activity_intensity))
    
    # Filter outliers
    outlier_mask = z_scores < outlier_threshold
    
    X_filtered = X[outlier_mask]
    y_p_filtered = y_p[outlier_mask]
    y_t_filtered = y_t[outlier_mask]
    y_c_filtered = y_c[outlier_mask]
    
    print(f"Removed {np.sum(~outlier_mask)} outliers ({np.sum(~outlier_mask)/len(X)*100:.1f}%)")
    print(f"Remaining samples: {len(X_filtered)}")
    
    return X_filtered, y_p_filtered, y_t_filtered, y_c_filtered

def improved_train_test_split(X, y_p, y_t, y_c, test_size=0.25, random_state=42):
    """
    Improved train/test split with better stratification
    
    Args:
        X: Sensor data
        y_p, y_t, y_c: Labels
        test_size: Test set size
        random_state: Random seed
    
    Returns:
        Improved train/test split
    """
    print("Creating improved train/test split...")
    
    # Filter outliers first
    X_clean, y_p_clean, y_t_clean, y_c_clean = filter_outliers_by_activity(X, y_p, y_t, y_c)
    
    # Convert to discrete for stratification
    y_p_discrete = (y_p_clean * 2).astype(int)
    
    # Use stratified split
    X_train, X_test, y_p_train, y_p_test, y_t_train, y_t_test, y_c_train, y_c_test = train_test_split(
        X_clean, y_p_clean, y_t_clean, y_c_clean,
        test_size=test_size, random_state=random_state, stratify=y_p_discrete
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    return X_train, X_test, y_p_train, y_p_test, y_t_train, y_t_test, y_c_train, y_c_test

def cross_validation_analysis(X, y_p, y_t, y_c, n_folds=5):
    """
    Run comprehensive cross-validation analysis
    
    Args:
        X: Sensor data
        y_p, y_t, y_c: Labels
        n_folds: Number of CV folds
    
    Returns:
        CV results DataFrame
    """
    print(f"Running {n_folds}-fold cross-validation...")
    
    # Filter outliers first
    X_clean, y_p_clean, y_t_clean, y_c_clean = filter_outliers_by_activity(X, y_p, y_t, y_c)
    
    # Convert to discrete for stratification
    y_p_discrete = (y_p_clean * 2).astype(int)
    
    # Use stratified K-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_clean, y_p_discrete)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Get fold data
        X_train_fold = X_clean[train_idx]
        X_test_fold = X_clean[test_idx]
        y_p_train_fold = y_p_clean[train_idx]
        y_p_test_fold = y_p_clean[test_idx]
        y_t_train_fold = y_t_clean[train_idx]
        y_t_test_fold = y_t_clean[test_idx]
        y_c_train_fold = y_c_clean[train_idx]
        y_c_test_fold = y_c_clean[test_idx]
        
        # Train model (simplified for CV)
        model = build_simple_cnn(X_train_fold.shape[1:], 3)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Convert labels to categorical
        from tensorflow.keras.utils import to_categorical
        y_p_train_cat = to_categorical(y_p_train_fold * 2, num_classes=3)
        y_p_test_cat = to_categorical(y_p_test_fold * 2, num_classes=3)
        
        # Train
        history = model.fit(
            X_train_fold, y_p_train_cat,
            validation_data=(X_test_fold, y_p_test_cat),
            epochs=30,
            batch_size=32,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test_fold, y_p_test_cat, verbose=0)
        
        cv_results.append({
            'fold': fold + 1,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'epochs_trained': len(history.history['loss'])
        })
        
        print(f"Fold {fold + 1} Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    
    # Analyze CV results
    cv_df = pd.DataFrame(cv_results)
    
    print(f"\n=== CROSS-VALIDATION RESULTS ===")
    print(f"Mean accuracy: {cv_df['test_accuracy'].mean():.4f} ± {cv_df['test_accuracy'].std():.4f}")
    print(f"Accuracy range: {cv_df['test_accuracy'].min():.4f} - {cv_df['test_accuracy'].max():.4f}")
    print(f"Range span: {(cv_df['test_accuracy'].max() - cv_df['test_accuracy'].min())*100:.1f}%")
    
    return cv_df

def build_simple_cnn(input_shape, n_classes):
    """Build simple CNN for cross-validation"""
    input_layer = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(n_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=output)
    return model

def run_deep_dive_analysis(X, y_p, y_t, y_c):
    """
    Run comprehensive deep dive analysis
    
    Args:
        X: Sensor data
        y_p, y_t, y_c: Labels
    
    Returns:
        Analysis results
    """
    print("=== DEEP DIVE ANALYSIS ===")
    
    # 1. Improved train/test split
    print("\n1. IMPROVED TRAIN/TEST SPLIT")
    X_train, X_test, y_p_train, y_p_test, y_t_train, y_t_test, y_c_train, y_c_test = improved_train_test_split(
        X, y_p, y_t, y_c
    )
    
    # 2. Cross-validation analysis
    print("\n2. CROSS-VALIDATION ANALYSIS")
    cv_results = cross_validation_analysis(X, y_p, y_t, y_c, n_folds=5)
    
    # 3. Performance comparison
    print("\n3. PERFORMANCE COMPARISON")
    print(f"CV Mean Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"CV Range: {(cv_results['test_accuracy'].max() - cv_results['test_accuracy'].min())*100:.1f}%")
    
    return {
        'cv_results': cv_results,
        'train_test_split': (X_train, X_test, y_p_train, y_p_test, y_t_train, y_t_test, y_c_train, y_c_test)
    }

if __name__ == "__main__":
    print("Deep Dive Analysis Module")
    print("Import this module to use the deep dive analysis functionality.")
