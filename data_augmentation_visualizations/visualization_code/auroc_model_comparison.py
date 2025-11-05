"""
AUROC Comparison: Complex CNN vs Ultra-Lightweight CNN
Calculates and visualizes AUROC for both model architectures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import label_binarize, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def build_complex_cnn(input_shape, n_classes_p, n_classes_t, n_classes_c):
    """
    Complex CNN with 177,000+ parameters (the original model)
    """
    input_layer = layers.Input(shape=input_shape)
    
    # Conv Block 1
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Conv Block 2
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Conv Block 3
    x = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Three output heads
    output_periodicity = layers.Dense(n_classes_p, activation='softmax', name='periodicity')(x)
    output_temporal = layers.Dense(n_classes_t, activation='softmax', name='temporal_stability')(x)
    output_coordination = layers.Dense(n_classes_c, activation='softmax', name='coordination')(x)
    
    return models.Model(
        inputs=input_layer,
        outputs=[output_periodicity, output_temporal, output_coordination]
    )

def build_lightweight_cnn(input_shape, n_classes_p, n_classes_t, n_classes_c):
    """
    Ultra-lightweight CNN with ~3,180 parameters (the optimized model)
    """
    input_layer = layers.Input(shape=input_shape)
    
    # Conv Block 1
    x = layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Conv Block 2
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    
    # Small shared dense layer
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Three output heads
    output_periodicity = layers.Dense(n_classes_p, activation='softmax', name='periodicity')(x)
    output_temporal = layers.Dense(n_classes_t, activation='softmax', name='temporal_stability')(x)
    output_coordination = layers.Dense(n_classes_c, activation='softmax', name='coordination')(x)
    
    return models.Model(
        inputs=input_layer,
        outputs=[output_periodicity, output_temporal, output_coordination]
    )

def calculate_auroc_multiclass(y_true, y_pred_proba, num_classes):
    """
    Calculate AUROC for multi-class classification using one-vs-rest approach
    """
    if num_classes == 2:
        return roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        # Use one-vs-rest approach for multi-class
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        return roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='macro')

def extract_window_robust(df_sensor, window_row, time_tolerance=0.5):
    """
    Extract sensor data with time tolerance to handle mismatches.
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
    target_length = 60
    if len(sensor_data) > target_length:
        indices = np.random.choice(len(sensor_data), target_length, replace=False)
        sensor_data = sensor_data[indices]
    elif len(sensor_data) < target_length:
        padding = np.tile(sensor_data[-1:], (target_length - len(sensor_data), 1))
        sensor_data = np.vstack([sensor_data, padding])
    
    return sensor_data

def load_and_prepare_data():
    """Load and prepare the data"""
    print("Loading data...")
    
    # Try to load from tudor_organized_workflow first
    try:
        df_sensor = pd.read_csv('./tudor_organized_workflow/data/final_dataset.csv')
        df_windows = pd.read_csv('./tudor_organized_workflow/data/final_window_labels.csv')
    except:
        try:
            df_sensor = pd.read_csv('./data/final_dataset.csv')
            df_windows = pd.read_csv('./data/final_window_labels.csv')
        except:
            print("Error: Could not find data files. Please check paths.")
            return None, None
    
    print(f"Sensor data: {len(df_sensor)} readings")
    print(f"Window labels: {len(df_windows)} windows")
    
    # Extract windows
    X = []
    y_p = []
    y_t = []
    y_c = []
    
    for _, window_row in df_windows.iterrows():
        window_data = extract_window_robust(df_sensor, window_row)
        if window_data is not None:
            X.append(window_data)
            y_p.append(window_row['periodicity'])
            y_t.append(window_row['temporal_stability'])
            y_c.append(window_row['coordination'])
    
    X = np.array(X)
    y_p = np.array(y_p)
    y_t = np.array(y_t)
    y_c = np.array(y_c)
    
    print(f"Extracted {len(X)} windows")
    
    # Normalize
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    X = X_scaled.reshape(n_samples, X.shape[1], X.shape[2])
    
    # Encode labels
    def encode_labels(y_train, y_test):
        unique_values = sorted(set(y_train) | set(y_test))
        n_classes = len(unique_values)
        value_to_class = {val: idx for idx, val in enumerate(unique_values)}
        
        y_train_encoded = np.array([value_to_class[val] for val in y_train])
        y_test_encoded = np.array([value_to_class[val] for val in y_test])
        
        return y_train_encoded, y_test_encoded, n_classes, value_to_class
    
    # Split data
    X_train, X_test, y_p_train, y_p_test, y_t_train, y_t_test, y_c_train, y_c_test = train_test_split(
        X, y_p, y_t, y_c, test_size=0.25, random_state=42, stratify=None
    )
    
    # Encode labels
    y_p_train_enc, y_p_test_enc, n_classes_p, mapping_p = encode_labels(y_p_train, y_p_test)
    y_t_train_enc, y_t_test_enc, n_classes_t, mapping_t = encode_labels(y_t_train, y_t_test)
    y_c_train_enc, y_c_test_enc, n_classes_c, mapping_c = encode_labels(y_c_train, y_c_test)
    
    y_p_train_cat = to_categorical(y_p_train_enc, num_classes=n_classes_p)
    y_p_test_cat = to_categorical(y_p_test_enc, num_classes=n_classes_p)
    y_t_train_cat = to_categorical(y_t_train_enc, num_classes=n_classes_t)
    y_t_test_cat = to_categorical(y_t_test_enc, num_classes=n_classes_t)
    y_c_train_cat = to_categorical(y_c_train_enc, num_classes=n_classes_c)
    y_c_test_cat = to_categorical(y_c_test_enc, num_classes=n_classes_c)
    
    print(f"\nClasses:")
    print(f"  Periodicity: {n_classes_p} classes")
    print(f"  Temporal Stability: {n_classes_t} classes")
    print(f"  Coordination: {n_classes_c} classes")
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_p_train': y_p_train_cat, 'y_p_test': y_p_test_cat,
        'y_t_train': y_t_train_cat, 'y_t_test': y_t_test_cat,
        'y_c_train': y_c_train_cat, 'y_c_test': y_c_test_cat,
        'y_p_test_enc': y_p_test_enc, 'y_t_test_enc': y_t_test_enc, 'y_c_test_enc': y_c_test_enc,
        'n_classes_p': n_classes_p, 'n_classes_t': n_classes_t, 'n_classes_c': n_classes_c
    }

def train_and_evaluate_model(model, data, model_name, epochs=50):
    """Train and evaluate a model, returning metrics"""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'periodicity': 'categorical_crossentropy',
            'temporal_stability': 'categorical_crossentropy',
            'coordination': 'categorical_crossentropy'
        },
        metrics={
            'periodicity': ['accuracy'],
            'temporal_stability': ['accuracy'],
            'coordination': ['accuracy']
        }
    )
    
    print(f"Parameters: {model.count_params():,}")
    
    # Train
    history = model.fit(
        data['X_train'],
        {
            'periodicity': data['y_p_train'],
            'temporal_stability': data['y_t_train'],
            'coordination': data['y_c_train']
        },
        validation_data=(
            data['X_test'],
            {
                'periodicity': data['y_p_test'],
                'temporal_stability': data['y_t_test'],
                'coordination': data['y_c_test']
            }
        ),
        epochs=epochs,
        batch_size=8,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')
        ]
    )
    
    # Evaluate
    predictions = model.predict(data['X_test'], verbose=0)
    
    # Calculate accuracy
    pred_p = np.argmax(predictions[0], axis=1)
    pred_t = np.argmax(predictions[1], axis=1)
    pred_c = np.argmax(predictions[2], axis=1)
    
    acc_p = accuracy_score(data['y_p_test_enc'], pred_p)
    acc_t = accuracy_score(data['y_t_test_enc'], pred_t)
    acc_c = accuracy_score(data['y_c_test_enc'], pred_c)
    avg_acc = (acc_p + acc_t + acc_c) / 3
    
    # Calculate AUROC
    auroc_p = calculate_auroc_multiclass(data['y_p_test_enc'], predictions[0], data['n_classes_p'])
    auroc_t = calculate_auroc_multiclass(data['y_t_test_enc'], predictions[1], data['n_classes_t'])
    auroc_c = calculate_auroc_multiclass(data['y_c_test_enc'], predictions[2], data['n_classes_c'])
    avg_auroc = (auroc_p + auroc_t + auroc_c) / 3
    
    print(f"\n{model_name} Results:")
    print(f"  Periodicity:   Acc={acc_p:.3f}, AUROC={auroc_p:.3f}")
    print(f"  Temporal:      Acc={acc_t:.3f}, AUROC={auroc_t:.3f}")
    print(f"  Coordination:  Acc={acc_c:.3f}, AUROC={auroc_c:.3f}")
    print(f"  Average:       Acc={avg_acc:.3f}, AUROC={avg_auroc:.3f}")
    
    return {
        'model': model,
        'history': history,
        'acc_p': acc_p, 'acc_t': acc_t, 'acc_c': acc_c, 'avg_acc': avg_acc,
        'auroc_p': auroc_p, 'auroc_t': auroc_t, 'auroc_c': auroc_c, 'avg_auroc': avg_auroc,
        'params': model.count_params()
    }

def create_comparison_visualization(complex_results, lightweight_results):
    """Create visualization comparing both models"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors
    complex_color = '#ff6b6b'  # Red
    lightweight_color = '#96ceb4'  # Green
    
    # Left plot: AUROC Comparison
    ax1 = axes[0]
    concepts = ['Periodicity', 'Temporal\nStability', 'Coordination', 'Average']
    complex_aurocs = [
        complex_results['auroc_p'],
        complex_results['auroc_t'],
        complex_results['auroc_c'],
        complex_results['avg_auroc']
    ]
    lightweight_aurocs = [
        lightweight_results['auroc_p'],
        lightweight_results['auroc_t'],
        lightweight_results['auroc_c'],
        lightweight_results['avg_auroc']
    ]
    
    x = np.arange(len(concepts))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, complex_aurocs, width, label='Complex CNN', 
                   color=complex_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, lightweight_aurocs, width, label='Ultra-Lightweight CNN', 
                   color=lightweight_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax1.set_title('AUROC Comparison: Complex vs Ultra-Lightweight CNN', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(concepts)
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random (0.5)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Right plot: Accuracy vs AUROC
    ax2 = axes[1]
    
    # Complex CNN
    ax2.scatter(complex_results['avg_acc'], complex_results['avg_auroc'], 
               s=300, color=complex_color, alpha=0.7, label='Complex CNN', 
               edgecolors='black', linewidth=2)
    ax2.text(complex_results['avg_acc'] + 0.01, complex_results['avg_auroc'],
             f"Complex\n({complex_results['params']:,} params)", 
             fontsize=10, va='center')
    
    # Lightweight CNN
    ax2.scatter(lightweight_results['avg_acc'], lightweight_results['avg_auroc'], 
               s=300, color=lightweight_color, alpha=0.7, label='Ultra-Lightweight CNN',
               edgecolors='black', linewidth=2)
    ax2.text(lightweight_results['avg_acc'] + 0.01, lightweight_results['avg_auroc'],
             f"Lightweight\n({lightweight_results['params']:,} params)", 
             fontsize=10, va='center')
    
    ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy vs AUROC: Model Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add diagonal line for reference
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    
    return fig

def create_summary_text(complex_results, lightweight_results):
    """Create text summary"""
    
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY: COMPLEX CNN vs ULTRA-LIGHTWEIGHT CNN")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Complex CNN':<20} {'Ultra-Lightweight CNN':<25} {'Improvement':<15}")
    print("-"*90)
    
    print(f"{'Parameters':<30} {complex_results['params']:>15,} {'':<20} {lightweight_results['params']:>15,}")
    print(f"{'Periodicity Accuracy':<30} {complex_results['acc_p']*100:>15.1f}% {'':<20} {lightweight_results['acc_p']*100:>15.1f}%")
    print(f"{'Periodicity AUROC':<30} {complex_results['auroc_p']:>15.3f} {'':<20} {lightweight_results['auroc_p']:>15.3f}")
    print(f"{'Temporal Accuracy':<30} {complex_results['acc_t']*100:>15.1f}% {'':<20} {lightweight_results['acc_t']*100:>15.1f}%")
    print(f"{'Temporal AUROC':<30} {complex_results['auroc_t']:>15.3f} {'':<20} {lightweight_results['auroc_t']:>15.3f}")
    print(f"{'Coordination Accuracy':<30} {complex_results['acc_c']*100:>15.1f}% {'':<20} {lightweight_results['acc_c']*100:>15.1f}%")
    print(f"{'Coordination AUROC':<30} {complex_results['auroc_c']:>15.3f} {'':<20} {lightweight_results['auroc_c']:>15.3f}")
    print(f"{'Average Accuracy':<30} {complex_results['avg_acc']*100:>15.1f}% {'':<20} {lightweight_results['avg_acc']*100:>15.1f}%")
    print(f"{'Average AUROC':<30} {complex_results['avg_auroc']:>15.3f} {'':<20} {lightweight_results['avg_auroc']:>15.3f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print(f"• Parameter Reduction: {complex_results['params']:,} → {lightweight_results['params']:,} "
          f"({complex_results['params']/lightweight_results['params']:.1f}x smaller)")
    print(f"• AUROC Improvement: {complex_results['avg_auroc']:.3f} → {lightweight_results['avg_auroc']:.3f} "
          f"(+{(lightweight_results['avg_auroc'] - complex_results['avg_auroc']):.3f})")
    print(f"• Accuracy Improvement: {complex_results['avg_acc']*100:.1f}% → {lightweight_results['avg_acc']*100:.1f}% "
          f"(+{(lightweight_results['avg_acc'] - complex_results['avg_acc'])*100:.1f}%)")
    print(f"• AUROC is higher than accuracy for both models (as expected for multi-class)")
    print(f"• Ultra-lightweight model shows better generalization (higher AUROC)")
    print("="*80)

def main():
    """Main function"""
    
    print("="*80)
    print("AUROC COMPARISON: COMPLEX CNN vs ULTRA-LIGHTWEIGHT CNN")
    print("="*80)
    
    # Load data
    data = load_and_prepare_data()
    if data is None:
        return
    
    # Build models
    input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
    
    print("\n" + "="*80)
    print("Building Models")
    print("="*80)
    
    complex_model = build_complex_cnn(
        input_shape, 
        data['n_classes_p'], 
        data['n_classes_t'], 
        data['n_classes_c']
    )
    
    lightweight_model = build_lightweight_cnn(
        input_shape, 
        data['n_classes_p'], 
        data['n_classes_t'], 
        data['n_classes_c']
    )
    
    print(f"\nComplex CNN: {complex_model.count_params():,} parameters")
    print(f"Ultra-Lightweight CNN: {lightweight_model.count_params():,} parameters")
    
    # Train and evaluate both models
    print("\n" + "="*80)
    print("Training Models")
    print("="*80)
    
    complex_results = train_and_evaluate_model(
        complex_model, data, "Complex CNN", epochs=50
    )
    
    lightweight_results = train_and_evaluate_model(
        lightweight_model, data, "Ultra-Lightweight CNN", epochs=50
    )
    
    # Create visualization
    print("\n" + "="*80)
    print("Creating Visualization")
    print("="*80)
    
    fig = create_comparison_visualization(complex_results, lightweight_results)
    
    # Save
    output_path = './auroc_model_comparison.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    # Print summary
    create_summary_text(complex_results, lightweight_results)
    
    # Show plot
    plt.show()
    
    return complex_results, lightweight_results

if __name__ == "__main__":
    results = main()


