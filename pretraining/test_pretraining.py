"""
Test script for the mask-and-reconstruct pretraining implementation.
This script runs a quick test to validate the implementation works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mask_reconstruct_pretraining import (
    SensorDataDataset, TimeSeriesEncoder, MaskReconstructTrainer,
    load_and_preprocess_data, create_data_loaders
)
import torch
import numpy as np
import matplotlib.pyplot as plt


def test_data_loading():
    """Test data loading and preprocessing."""
    print("Testing data loading...")
    
    # Create dummy data for testing
    dummy_data = np.random.randn(1000, 3)  # 1000 samples, 3 features
    
    # Test dataset creation
    dataset = SensorDataDataset(dummy_data, window_size=64, mask_ratio=0.15)
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test getting a sample
    sample = dataset[0]
    print(f"Sample input shape: {sample['input'].shape}")
    print(f"Sample target shape: {sample['target'].shape}")
    print(f"Sample mask shape: {sample['mask'].shape}")
    print(f"Mask ratio: {sample['mask'].sum().item() / len(sample['mask']):.3f}")
    
    return True


def test_model_architecture():
    """Test the model architecture."""
    print("\nTesting model architecture...")
    
    model = TimeSeriesEncoder(input_dim=3, hidden_dim=64, num_layers=3)
    
    # Test forward pass
    batch_size = 4
    seq_len = 64
    input_tensor = torch.randn(batch_size, seq_len, 3)
    
    with torch.no_grad():
        encoded, reconstructed = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Check output shapes
    assert encoded.shape[0] == batch_size, "Batch size mismatch in encoded output"
    assert reconstructed.shape == (batch_size, 3), "Reconstructed shape mismatch"
    
    print("Model architecture test passed!")
    return True


def test_training_step():
    """Test a single training step."""
    print("\nTesting training step...")
    
    # Create dummy data
    dummy_data = np.random.randn(500, 3)
    dataset = SensorDataDataset(dummy_data, window_size=32, mask_ratio=0.15)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create model and trainer
    model = TimeSeriesEncoder(input_dim=3, hidden_dim=32, num_layers=2)
    trainer = MaskReconstructTrainer(model, device='cpu')
    trainer.setup_optimizer(learning_rate=1e-3)
    
    # Test one training step
    train_loss = trainer.train_epoch(dataloader)
    print(f"Training loss: {train_loss:.4f}")
    
    # Test validation
    val_loss = trainer.validate(dataloader)
    print(f"Validation loss: {val_loss:.4f}")
    
    print("Training step test passed!")
    return True


def test_with_real_data_sample():
    """Test with a small sample of real data."""
    print("\nTesting with real data sample...")
    
    try:
        # Load a small sample of real data
        import pandas as pd
        df = pd.read_csv('/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/dataset.csv', nrows=1000)
        sensor_data = df[['x-axis', 'y-axis', 'z-axis']].values
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        sensor_data = scaler.fit_transform(sensor_data)
        
        # Create dataset and dataloader
        dataset = SensorDataDataset(sensor_data, window_size=32, mask_ratio=0.15)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Create model and trainer
        model = TimeSeriesEncoder(input_dim=3, hidden_dim=32, num_layers=2)
        trainer = MaskReconstructTrainer(model, device='cpu')
        trainer.setup_optimizer(learning_rate=1e-3)
        
        # Test one epoch
        train_loss = trainer.train_epoch(dataloader)
        print(f"Real data training loss: {train_loss:.4f}")
        
        print("Real data test passed!")
        return True
        
    except Exception as e:
        print(f"Real data test failed: {e}")
        return False


def visualize_masking():
    """Visualize the masking process."""
    print("\nVisualizing masking process...")
    
    # Create sample data
    t = np.linspace(0, 4*np.pi, 64)
    x = np.sin(t)
    y = np.cos(t)
    z = np.sin(2*t)
    sample_data = np.column_stack([x, y, z])
    
    # Create dataset
    dataset = SensorDataDataset(sample_data, window_size=64, mask_ratio=0.2)
    sample = dataset[0]
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(sample['target'][:, 0].numpy(), label='Original X', alpha=0.7)
    axes[0].plot(sample['input'][:, 0].numpy(), label='Masked X', alpha=0.7)
    axes[0].set_title('X-axis')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(sample['target'][:, 1].numpy(), label='Original Y', alpha=0.7)
    axes[1].plot(sample['input'][:, 1].numpy(), label='Masked Y', alpha=0.7)
    axes[1].set_title('Y-axis')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(sample['target'][:, 2].numpy(), label='Original Z', alpha=0.7)
    axes[2].plot(sample['input'][:, 2].numpy(), label='Masked Z', alpha=0.7)
    axes[2].set_title('Z-axis')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining/masking_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Masking visualization saved!")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("TESTING MASK-AND-RECONSTRUCT PRETRAINING")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Architecture", test_model_architecture),
        ("Training Step", test_training_step),
        ("Real Data Sample", test_with_real_data_sample),
        ("Masking Visualization", visualize_masking)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name}: FAILED - {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()

