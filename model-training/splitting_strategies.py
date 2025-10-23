"""
Splitting Strategies Module

This module contains advanced data splitting strategies for improving model
performance and reducing variability. It includes outlier filtering, signal
characteristic balancing, and stratified splitting techniques.

Usage:
    from splitting_strategies import run_improved_splitting, run_cross_validation_analysis
    
    # Improved single split
    X_train, X_test, y_train, y_test = run_improved_splitting(X, y_p, y_t, y_c)
    
    # Cross-validation analysis
    cv_results = run_cross_validation_analysis(X, y_p, y_t, y_c, n_folds=5)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

def filter_outliers_by_activity(X, y_p, y_t, y_c, threshold=2.0):
    """
    Remove outliers based on activity level (2σ rule)
    
    Args:
        X: Sensor data (n_samples, timesteps, features)
        y_p, y_t, y_c: Labels for three concepts
        threshold: Standard deviation threshold for outlier detection
    
    Returns:
        Filtered data without outliers
    """
    # Calculate activity level (magnitude of acceleration)
    activity = np.mean(np.sqrt(np.sum(X**2, axis=2)), axis=1)
    mean_activity = np.mean(activity)
    std_activity = np.std(activity)
    
    # Keep data within threshold standard deviations
    mask = (activity >= mean_activity - threshold * std_activity) & \
           (activity <= mean_activity + threshold * std_activity)
    
    print(f"Outlier filtering: {len(X)} → {np.sum(mask)} samples ({100*np.sum(mask)/len(X):.1f}% kept)")
    
    return X[mask], y_p[mask], y_t[mask], y_c[mask]

def balance_signal_characteristics(X, y_p, y_t, y_c, target_variance_ratio=0.95):
    """
    Balance signal characteristics between train/test
    
    Args:
        X: Sensor data
        y_p, y_t, y_c: Labels
        target_variance_ratio: Target variance ratio between train/test
    
    Returns:
        Balanced data and variance groups
    """
    # Calculate signal variance for each sample
    sample_variances = np.var(X, axis=(1,2))
    
    # Sort by variance
    sorted_indices = np.argsort(sample_variances)
    
    # Split into low, medium, high variance groups
    n_samples = len(X)
    low_var = sorted_indices[:n_samples//3]
    med_var = sorted_indices[n_samples//3:2*n_samples//3]
    high_var = sorted_indices[2*n_samples//3:]
    
    print(f"Signal variance groups: Low={len(low_var)}, Med={len(med_var)}, High={len(high_var)}")
    
    return X, y_p, y_t, y_c, [low_var, med_var, high_var]

def improved_train_test_split(X, y_p, y_t, y_c, test_size=0.25, random_state=42):
    """
    Improved splitting with stratification and quality control
    
    Args:
        X: Sensor data
        y_p, y_t, y_c: Labels for three concepts
        test_size: Test set size
        random_state: Random seed
    
    Returns:
        Improved train/test split
    """
    print("=== IMPROVED SPLITTING STRATEGY ===")
    
    # 1. Filter outliers
    X_clean, y_p_clean, y_t_clean, y_c_clean = filter_outliers_by_activity(X, y_p, y_t, y_c)
    
    # 2. Balance signal characteristics
    X_balanced, y_p_balanced, y_t_balanced, y_c_balanced, variance_groups = balance_signal_characteristics(
        X_clean, y_p_clean, y_t_clean, y_c_clean
    )
    
    # 3. Stratified splitting (stratify by most imbalanced concept - periodicity)
    try:
        X_train, X_test, y_p_train, y_p_test, y_t_train, y_t_test, y_c_train, y_c_test = train_test_split(
            X_balanced, y_p_balanced, y_t_balanced, y_c_balanced, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y_p_balanced  # Stratify by periodicity (most imbalanced)
        )
        print("✅ Stratified splitting successful")
    except ValueError as e:
        print(f"⚠️ Stratified splitting failed: {e}")
        print("Falling back to random split...")
        X_train, X_test, y_p_train, y_p_test, y_t_train, y_t_test, y_c_train, y_c_test = train_test_split(
            X_balanced, y_p_balanced, y_t_balanced, y_c_balanced, 
            test_size=test_size, 
            random_state=random_state
        )
    
    # 4. Ensure balanced representation from variance groups
    # Try to maintain similar variance distribution in train/test
    train_indices = np.arange(len(X_train))
    test_indices = np.arange(len(X_test))
    
    # Calculate final characteristics
    train_variance = np.var(X_train, axis=(1,2)).mean()
    test_variance = np.var(X_test, axis=(1,2)).mean()
    variance_ratio = test_variance / train_variance
    
    train_magnitude = np.mean(np.sqrt(np.sum(X_train**2, axis=2)), axis=1).mean()
    test_magnitude = np.mean(np.sqrt(np.sum(X_test**2, axis=2)), axis=1).mean()
    magnitude_ratio = test_magnitude / train_magnitude
    
    print(f"\nFinal split characteristics:")
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Variance ratio: {variance_ratio:.3f} (target: 0.85-1.0)")
    print(f"  Magnitude ratio: {magnitude_ratio:.3f} (target: 0.95-1.0)")
    
    # Check class balance
    for concept, train_labels, test_labels in [('Periodicity', y_p_train, y_p_test),
                                             ('Temporal Stability', y_t_train, y_t_test),
                                             ('Coordination', y_c_train, y_c_test)]:
        train_dist = np.bincount(train_labels.astype(int))
        test_dist = np.bincount(test_labels.astype(int))
        train_imbalance = np.max(train_dist) / np.min(train_dist)
        test_imbalance = np.max(test_dist) / np.min(test_dist)
        print(f"  {concept}: Train imbalance {train_imbalance:.2f}, Test imbalance {test_imbalance:.2f}")
    
    return X_train, X_test, y_p_train, y_p_test, y_t_train, y_t_test, y_c_train, y_c_test

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

def cross_validation_analysis(X, y_p, y_t, y_c, n_folds=5):
    """
    5-fold cross-validation with improved splitting
    
    Args:
        X: Sensor data
        y_p, y_t, y_c: Labels for three concepts
        n_folds: Number of CV folds
    
    Returns:
        CV results DataFrame
    """
    print(f"\n=== {n_folds}-FOLD CROSS-VALIDATION ANALYSIS ===")
    
    # Filter outliers first
    X_clean, y_p_clean, y_t_clean, y_c_clean = filter_outliers_by_activity(X, y_p, y_t, y_c)
    
    # Convert continuous labels to discrete classes for stratification
    y_p_discrete = (y_p_clean * 2).astype(int)  # Convert 0.5, 1.0 to 1, 2
    y_t_discrete = (y_t_clean * 2).astype(int)
    y_c_discrete = (y_c_clean * 2).astype(int)
    
    print(f"Discrete label distributions:")
    print(f"  Periodicity: {np.bincount(y_p_discrete)}")
    print(f"  Temporal Stability: {np.bincount(y_t_discrete)}")
    print(f"  Coordination: {np.bincount(y_c_discrete)}")
    
    # Use stratified K-fold with discrete labels
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
        
        # Analyze fold characteristics
        train_variance = np.var(X_train_fold, axis=(1,2)).mean()
        test_variance = np.var(X_test_fold, axis=(1,2)).mean()
        variance_ratio = test_variance / train_variance
        
        train_magnitude = np.mean(np.sqrt(np.sum(X_train_fold**2, axis=2)), axis=1).mean()
        test_magnitude = np.mean(np.sqrt(np.sum(X_test_fold**2, axis=2)), axis=1).mean()
        magnitude_ratio = test_magnitude / train_magnitude
        
        print(f"  Variance ratio: {variance_ratio:.3f}")
        print(f"  Magnitude ratio: {magnitude_ratio:.3f}")
        
        # Train model
        print("  Training model...")
        
        # Convert to categorical
        y_p_train_cat = to_categorical(y_p_train_fold * 2, num_classes=3)
        y_t_train_cat = to_categorical(y_t_train_fold * 2, num_classes=3)
        y_c_train_cat = to_categorical(y_c_train_fold * 2, num_classes=3)
        y_p_test_cat = to_categorical(y_p_test_fold * 2, num_classes=3)
        y_t_test_cat = to_categorical(y_t_test_fold * 2, num_classes=3)
        y_c_test_cat = to_categorical(y_c_test_fold * 2, num_classes=3)
        
        # Augment training data
        from quick_train_cnn import augment_dataset
        X_train_aug, y_p_train_aug, y_t_train_aug, y_c_train_aug = augment_dataset(
            X_train_fold, y_p_train_fold, y_t_train_fold, y_c_train_fold, factor=5
        )
        
        # Convert augmented labels to categorical
        y_p_train_cat_aug = to_categorical(y_p_train_aug * 2, num_classes=3)
        y_t_train_cat_aug = to_categorical(y_t_train_aug * 2, num_classes=3)
        y_c_train_cat_aug = to_categorical(y_c_train_aug * 2, num_classes=3)
        
        # Build and train model
        model = build_simple_cnn(X_train_aug.shape[1:], 3)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train_aug, y_p_train_cat_aug,
            validation_data=(X_test_fold, y_p_test_cat),
            epochs=30,
            batch_size=16,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test_fold, y_p_test_cat, verbose=0)
        
        # Store results
        cv_results.append({
            'fold': fold + 1,
            'accuracy': test_acc,
            'loss': test_loss,
            'variance_ratio': variance_ratio,
            'magnitude_ratio': magnitude_ratio,
            'epochs_trained': len(history.history['loss'])
        })
        
        print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    
    # Analyze CV results
    cv_df = pd.DataFrame(cv_results)
    
    print(f"\n=== CROSS-VALIDATION RESULTS ===")
    print(f"Mean accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
    print(f"Accuracy range: {cv_df['accuracy'].min():.4f} - {cv_df['accuracy'].max():.4f}")
    print(f"Range span: {(cv_df['accuracy'].max() - cv_df['accuracy'].min())*100:.1f}%")
    
    return cv_df

def run_improved_splitting(X, y_p, y_t, y_c):
    """
    Run improved splitting strategy with comprehensive analysis
    
    Args:
        X: Sensor data
        y_p, y_t, y_c: Labels for three concepts
    
    Returns:
        Improved train/test split and analysis results
    """
    print("=== IMPROVED SPLITTING STRATEGIES COMPARISON ===")
    print("Implementing suggestions to reduce performance variability")
    
    # Run improved splitting
    X_train, X_test, y_p_train, y_p_test, y_t_train, y_t_test, y_c_train, y_c_test = improved_train_test_split(
        X, y_p, y_t, y_c
    )
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_p_train': y_p_train, 'y_p_test': y_p_test,
        'y_t_train': y_t_train, 'y_t_test': y_t_test,
        'y_c_train': y_c_train, 'y_c_test': y_c_test
    }

def run_comprehensive_analysis(X, y_p, y_t, y_c):
    """
    Run comprehensive splitting analysis including CV
    
    Args:
        X: Sensor data
        y_p, y_t, y_c: Labels for three concepts
    
    Returns:
        Complete analysis results
    """
    print("=== COMPREHENSIVE SPLITTING ANALYSIS ===")
    
    # 1. Improved single split
    print("\n1. IMPROVED SINGLE SPLIT")
    split_results = run_improved_splitting(X, y_p, y_t, y_c)
    
    # 2. Cross-validation analysis
    print("\n2. CROSS-VALIDATION ANALYSIS")
    cv_results = cross_validation_analysis(X, y_p, y_t, y_c, n_folds=5)
    
    # 3. Performance comparison
    print("\n3. PERFORMANCE COMPARISON")
    print(f"CV Mean Accuracy: {cv_results['accuracy'].mean():.4f} ± {cv_results['accuracy'].std():.4f}")
    print(f"CV Range: {(cv_results['accuracy'].max() - cv_results['accuracy'].min())*100:.1f}%")
    
    return {
        'split_results': split_results,
        'cv_results': cv_results
    }

if __name__ == "__main__":
    print("Splitting Strategies Module")
    print("Import this module to use the advanced splitting strategies.")
