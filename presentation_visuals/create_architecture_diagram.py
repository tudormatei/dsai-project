"""
Create ultra-lightweight CNN architecture diagram for presentation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

def create_architecture_diagram():
    """Create a detailed architecture diagram of the ultra-lightweight CNN"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    input_color = '#E3F2FD'  # Light blue
    conv_color = '#BBDEFB'   # Medium blue
    pool_color = '#90CAF9'   # Darker blue
    dense_color = '#64B5F6'  # Even darker blue
    output_color = '#42A5F5' # Darkest blue
    text_color = '#1565C0'   # Dark blue text
    
    # Title
    ax.text(5, 11.5, 'Ultra-Lightweight CNN Architecture', 
            fontsize=20, fontweight='bold', ha='center', color='#0D47A1')
    ax.text(5, 11, '3,180 Parameters | 100 Labeled Windows | 70.7% Accuracy', 
            fontsize=14, ha='center', color='#424242')
    
    # Input Layer
    input_box = FancyBboxPatch((0.5, 9.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=input_color, 
                               edgecolor=text_color, linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 10, 'Input\n(60, 3)', ha='center', va='center', 
            fontsize=12, fontweight='bold', color=text_color)
    ax.text(1.5, 9.7, '60 timesteps\n3 sensor axes', ha='center', va='center', 
            fontsize=10, color=text_color)
    
    # Arrow 1
    ax.arrow(2.6, 10, 0.8, 0, head_width=0.1, head_length=0.1, 
             fc=text_color, ec=text_color, linewidth=2)
    
    # Conv1D Layer 1
    conv1_box = FancyBboxPatch((3.5, 9.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=conv_color, 
                               edgecolor=text_color, linewidth=2)
    ax.add_patch(conv1_box)
    ax.text(4.5, 10, 'Conv1D\n16 filters', ha='center', va='center', 
            fontsize=12, fontweight='bold', color=text_color)
    ax.text(4.5, 9.7, 'kernel=3, padding=same\n160 parameters', ha='center', va='center', 
            fontsize=10, color=text_color)
    
    # Arrow 2
    ax.arrow(5.6, 10, 0.8, 0, head_width=0.1, head_length=0.1, 
             fc=text_color, ec=text_color, linewidth=2)
    
    # MaxPool1D
    pool1_box = FancyBboxPatch((6.5, 9.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=pool_color, 
                               edgecolor=text_color, linewidth=2)
    ax.add_patch(pool1_box)
    ax.text(7.5, 10, 'MaxPool1D\npool_size=2', ha='center', va='center', 
            fontsize=12, fontweight='bold', color=text_color)
    ax.text(7.5, 9.7, 'Output: (30, 16)', ha='center', va='center', 
            fontsize=10, color=text_color)
    
    # Arrow 3 (down)
    ax.arrow(7.5, 9.4, 0, -0.8, head_width=0.1, head_length=0.1, 
             fc=text_color, ec=text_color, linewidth=2)
    
    # Conv1D Layer 2
    conv2_box = FancyBboxPatch((6.5, 7.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=conv_color, 
                               edgecolor=text_color, linewidth=2)
    ax.add_patch(conv2_box)
    ax.text(7.5, 8, 'Conv1D\n32 filters', ha='center', va='center', 
            fontsize=12, fontweight='bold', color=text_color)
    ax.text(7.5, 7.7, 'kernel=3, padding=same\n1,568 parameters', ha='center', va='center', 
            fontsize=10, color=text_color)
    
    # Arrow 4 (down)
    ax.arrow(7.5, 7.4, 0, -0.8, head_width=0.1, head_length=0.1, 
             fc=text_color, ec=text_color, linewidth=2)
    
    # Global Average Pooling
    gap_box = FancyBboxPatch((6.5, 5.5), 2, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=pool_color, 
                             edgecolor=text_color, linewidth=2)
    ax.add_patch(gap_box)
    ax.text(7.5, 6, 'GlobalAvgPool1D', ha='center', va='center', 
            fontsize=12, fontweight='bold', color=text_color)
    ax.text(7.5, 5.7, 'Output: (32,)', ha='center', va='center', 
            fontsize=10, color=text_color)
    
    # Arrow 5 (down)
    ax.arrow(7.5, 5.4, 0, -0.8, head_width=0.1, head_length=0.1, 
             fc=text_color, ec=text_color, linewidth=2)
    
    # Dense Layer
    dense_box = FancyBboxPatch((6.5, 3.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=dense_color, 
                               edgecolor=text_color, linewidth=2)
    ax.add_patch(dense_box)
    ax.text(7.5, 4, 'Dense\n32 units', ha='center', va='center', 
            fontsize=12, fontweight='bold', color=text_color)
    ax.text(7.5, 3.7, '1,056 parameters', ha='center', va='center', 
            fontsize=10, color=text_color)
    
    # Arrow 6 (down)
    ax.arrow(7.5, 3.4, 0, -0.8, head_width=0.1, head_length=0.1, 
             fc=text_color, ec=text_color, linewidth=2)
    
    # Output Heads
    # Periodicity
    period_box = FancyBboxPatch((0.5, 1.5), 1.8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=output_color, 
                                edgecolor=text_color, linewidth=2)
    ax.add_patch(period_box)
    ax.text(1.4, 2, 'Periodicity', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(1.4, 1.7, '3 classes\n99 params', ha='center', va='center', 
            fontsize=9, color='white')
    
    # Temporal Stability
    temp_box = FancyBboxPatch((2.5, 1.5), 1.8, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=output_color, 
                              edgecolor=text_color, linewidth=2)
    ax.add_patch(temp_box)
    ax.text(3.4, 2, 'Temporal\nStability', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(3.4, 1.7, '3 classes\n99 params', ha='center', va='center', 
            fontsize=9, color='white')
    
    # Coordination
    coord_box = FancyBboxPatch((4.5, 1.5), 1.8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=output_color, 
                               edgecolor=text_color, linewidth=2)
    ax.add_patch(coord_box)
    ax.text(5.4, 2, 'Coordination', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(5.4, 1.7, '3 classes\n99 params', ha='center', va='center', 
            fontsize=9, color='white')
    
    # Arrows to outputs
    ax.arrow(7.5, 3.4, -2.1, -1.6, head_width=0.1, head_length=0.1, 
             fc=text_color, ec=text_color, linewidth=2)
    ax.arrow(7.5, 3.4, -1.1, -1.6, head_width=0.1, head_length=0.1, 
             fc=text_color, ec=text_color, linewidth=2)
    ax.arrow(7.5, 3.4, -0.1, -1.6, head_width=0.1, head_length=0.1, 
             fc=text_color, ec=text_color, linewidth=2)
    
    # Dropout indicators
    ax.text(3.5, 8.5, 'Dropout 0.2', ha='center', va='center', 
            fontsize=9, color='#FF5722', fontweight='bold')
    ax.text(7.5, 6.5, 'Dropout 0.3', ha='center', va='center', 
            fontsize=9, color='#FF5722', fontweight='bold')
    ax.text(7.5, 4.5, 'Dropout 0.3', ha='center', va='center', 
            fontsize=9, color='#FF5722', fontweight='bold')
    
    # Key insights box
    insight_box = FancyBboxPatch((8.5, 7), 1.4, 4, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#FFF3E0', 
                                 edgecolor='#FF9800', linewidth=2)
    ax.add_patch(insight_box)
    ax.text(9.2, 10.5, 'Key Insights', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#E65100')
    ax.text(9.2, 9.8, '• Right-sized for\n  small dataset', ha='center', va='center', 
            fontsize=10, color='#E65100')
    ax.text(9.2, 9.2, '• Multi-task\n  learning', ha='center', va='center', 
            fontsize=10, color='#E65100')
    ax.text(9.2, 8.6, '• Domain-specific\n  design', ha='center', va='center', 
            fontsize=10, color='#E65100')
    ax.text(9.2, 8.0, '• 55x smaller than\n  original model', ha='center', va='center', 
            fontsize=10, color='#E65100')
    ax.text(9.2, 7.4, '• 2.7x better\n  accuracy', ha='center', va='center', 
            fontsize=10, color='#E65100')
    
    # Performance metrics
    ax.text(1.5, 0.5, 'Performance Metrics:', fontsize=12, fontweight='bold', color='#0D47A1')
    ax.text(1.5, 0.2, '• Periodicity: 88.0% accuracy', fontsize=10, color='#424242')
    ax.text(4.5, 0.2, '• Temporal Stability: 56.0% accuracy', fontsize=10, color='#424242')
    ax.text(7.5, 0.2, '• Coordination: 68.0% accuracy', fontsize=10, color='#424242')
    
    plt.tight_layout()
    plt.savefig('/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/presentation_visuals/ultra_lightweight_cnn_architecture.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    create_architecture_diagram()


