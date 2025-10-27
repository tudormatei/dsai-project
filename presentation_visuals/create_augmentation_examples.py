"""
Create data augmentation examples showing realistic transformations
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

def create_augmentation_examples():
    """Create examples of data augmentation techniques applied to sensor data"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Generate sample sensor data (3-axis accelerometer)
    t = np.linspace(0, 3, 60)  # 3 seconds, 20Hz sampling
    np.random.seed(42)
    
    # Original signal (walking pattern)
    x_orig = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(60)
    y_orig = 0.3 * np.sin(2 * np.pi * 1.2 * t + np.pi/4) + 0.1 * np.random.randn(60)
    z_orig = 0.8 + 0.2 * np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(60)
    
    # === JITTERING AUGMENTATION ===
    ax1.set_title('Jittering Augmentation', fontsize=14, fontweight='bold', color='#1976D2')
    ax1.plot(t, x_orig, 'b-', linewidth=2, label='X-axis (Original)')
    ax1.plot(t, y_orig, 'g-', linewidth=2, label='Y-axis (Original)')
    ax1.plot(t, z_orig, 'r-', linewidth=2, label='Z-axis (Original)')
    
    # Add jittered versions
    noise_std = 0.03
    x_jitter = x_orig + noise_std * np.random.randn(60)
    y_jitter = y_orig + noise_std * np.random.randn(60)
    z_jitter = z_orig + noise_std * np.random.randn(60)
    
    ax1.plot(t, x_jitter, 'b--', linewidth=1.5, alpha=0.7, label='X-axis (Jittered)')
    ax1.plot(t, y_jitter, 'g--', linewidth=1.5, alpha=0.7, label='Y-axis (Jittered)')
    ax1.plot(t, z_jitter, 'r--', linewidth=1.5, alpha=0.7, label='Z-axis (Jittered)')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Acceleration (g)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.1, 0.95, 'Adds Gaussian noise (σ=0.03)\nSimulates sensor imperfections', 
             transform=ax1.transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # === SCALING AUGMENTATION ===
    ax2.set_title('Scaling Augmentation', fontsize=14, fontweight='bold', color='#388E3C')
    ax2.plot(t, x_orig, 'b-', linewidth=2, label='X-axis (Original)')
    ax2.plot(t, y_orig, 'g-', linewidth=2, label='Y-axis (Original)')
    ax2.plot(t, z_orig, 'r-', linewidth=2, label='Z-axis (Original)')
    
    # Add scaled versions
    scale_factor = 1.05
    x_scale = x_orig * scale_factor
    y_scale = y_orig * scale_factor
    z_scale = z_orig * scale_factor
    
    ax2.plot(t, x_scale, 'b--', linewidth=1.5, alpha=0.7, label='X-axis (Scaled)')
    ax2.plot(t, y_scale, 'g--', linewidth=1.5, alpha=0.7, label='Y-axis (Scaled)')
    ax2.plot(t, z_scale, 'r--', linewidth=1.5, alpha=0.7, label='Z-axis (Scaled)')
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Acceleration (g)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.1, 0.95, 'Multiplies by 0.95-1.05\nSimulates speed variations', 
             transform=ax2.transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # === ROTATION AUGMENTATION ===
    ax3.set_title('Rotation Augmentation', fontsize=14, fontweight='bold', color='#7B1FA2')
    
    # Original 3D data
    data_orig = np.column_stack([x_orig, y_orig, z_orig])
    
    # Rotation matrix (30 degrees around Z-axis)
    angle = np.pi/6  # 30 degrees
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    data_rotated = data_orig @ rotation_matrix.T
    
    ax3.plot(t, data_orig[:, 0], 'b-', linewidth=2, label='X-axis (Original)')
    ax3.plot(t, data_orig[:, 1], 'g-', linewidth=2, label='Y-axis (Original)')
    ax3.plot(t, data_orig[:, 2], 'r-', linewidth=2, label='Z-axis (Original)')
    
    ax3.plot(t, data_rotated[:, 0], 'b--', linewidth=1.5, alpha=0.7, label='X-axis (Rotated)')
    ax3.plot(t, data_rotated[:, 1], 'g--', linewidth=1.5, alpha=0.7, label='Y-axis (Rotated)')
    ax3.plot(t, data_rotated[:, 2], 'r--', linewidth=1.5, alpha=0.7, label='Z-axis (Rotated)')
    
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Acceleration (g)')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.1, 0.95, '3D rotation (±30°)\nSimulates device orientation', 
             transform=ax3.transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    
    # === COMBINED AUGMENTATION IMPACT ===
    ax4.set_title('Combined Augmentation Impact', fontsize=14, fontweight='bold', color='#D32F2F')
    
    # Show the effect on dataset size
    strategies = ['Original', 'Jittering', 'Scaling', 'Rotation', 'All Combined']
    samples = [73, 146, 219, 292, 803]  # 2x, 3x, 4x, 11x augmentation
    colors = ['#FFCDD2', '#FFE0B2', '#FFF3E0', '#E8F5E8', '#C8E6C9']
    
    bars = ax4.bar(strategies, samples, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, sample in zip(bars, samples):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{sample}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_ylabel('Number of Training Samples')
    ax4.set_ylim(0, 900)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add performance improvement text
    ax4.text(0.5, 0.8, 'Performance Improvement:', 
             transform=ax4.transAxes, fontsize=12, fontweight='bold')
    ax4.text(0.5, 0.7, '49.3% → 70.7% accuracy', 
             transform=ax4.transAxes, fontsize=11, color='#2E7D32')
    ax4.text(0.5, 0.6, '+21.4% improvement', 
             transform=ax4.transAxes, fontsize=11, color='#2E7D32')
    ax4.text(0.5, 0.5, '10.7x more training data', 
             transform=ax4.transAxes, fontsize=11, color='#2E7D32')
    
    # Add concept labels preservation note
    ax4.text(0.5, 0.3, 'Key: All transformations preserve concept labels!', 
             transform=ax4.transAxes, fontsize=10, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Overall title
    fig.suptitle('Data Augmentation Techniques for Sensor Data', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.savefig('/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/presentation_visuals/data_augmentation_examples.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    create_augmentation_examples()
