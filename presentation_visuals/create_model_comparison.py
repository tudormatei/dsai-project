"""
Create before/after comparison of model complexity
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

def create_model_comparison():
    """Create a side-by-side comparison of the old vs new model architecture"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Colors
    old_color = '#FFCDD2'  # Light red
    new_color = '#E8F5E8'  # Light green
    text_color = '#424242'
    
    # === OLD MODEL (LEFT SIDE) ===
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    
    # Title
    ax1.text(5, 11.5, 'BEFORE: Complex CNN', 
             fontsize=18, fontweight='bold', ha='center', color='#D32F2F')
    ax1.text(5, 11, '177,000+ Parameters | 26% Accuracy', 
             fontsize=14, ha='center', color='#424242')
    
    # Old model architecture (simplified)
    # Input
    input_box = FancyBboxPatch((1, 9), 8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=old_color, 
                               edgecolor='#D32F2F', linewidth=2)
    ax1.add_patch(input_box)
    ax1.text(5, 9.5, 'Input (60, 3)', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    # Conv layers
    conv1_box = FancyBboxPatch((1, 7.5), 8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=old_color, 
                               edgecolor='#D32F2F', linewidth=2)
    ax1.add_patch(conv1_box)
    ax1.text(5, 8, 'Conv1D: 64 filters', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    conv2_box = FancyBboxPatch((1, 6), 8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=old_color, 
                               edgecolor='#D32F2F', linewidth=2)
    ax1.add_patch(conv2_box)
    ax1.text(5, 6.5, 'Conv1D: 128 filters', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    conv3_box = FancyBboxPatch((1, 4.5), 8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=old_color, 
                               edgecolor='#D32F2F', linewidth=2)
    ax1.add_patch(conv3_box)
    ax1.text(5, 5, 'Conv1D: 256 filters', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    # Dense layers
    dense1_box = FancyBboxPatch((1, 3), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=old_color, 
                                edgecolor='#D32F2F', linewidth=2)
    ax1.add_patch(dense1_box)
    ax1.text(5, 3.5, 'Dense: 128 units', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    dense2_box = FancyBboxPatch((1, 1.5), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=old_color, 
                                edgecolor='#D32F2F', linewidth=2)
    ax1.add_patch(dense2_box)
    ax1.text(5, 2, 'Dense: 64 units', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    # Output
    output_box = FancyBboxPatch((1, 0.5), 8, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=old_color, 
                                edgecolor='#D32F2F', linewidth=2)
    ax1.add_patch(output_box)
    ax1.text(5, 0.9, '3 Output Heads', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    # Problems box
    problems_box = FancyBboxPatch((0.2, 0.2), 9.6, 1.2, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#FFEBEE', 
                                  edgecolor='#D32F2F', linewidth=2)
    ax1.add_patch(problems_box)
    ax1.text(5, 0.8, 'PROBLEMS:', ha='center', va='center', 
             fontsize=12, fontweight='bold', color='#D32F2F')
    ax1.text(5, 0.5, '• 2,360 parameters per training sample', ha='center', va='center', 
             fontsize=10, color='#D32F2F')
    ax1.text(5, 0.2, '• Severe overfitting - memorizes, not learns', ha='center', va='center', 
             fontsize=10, color='#D32F2F')
    
    # === NEW MODEL (RIGHT SIDE) ===
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    
    # Title
    ax2.text(5, 11.5, 'AFTER: Ultra-Lightweight CNN', 
             fontsize=18, fontweight='bold', ha='center', color='#2E7D32')
    ax2.text(5, 11, '3,180 Parameters | 70.7% Accuracy', 
             fontsize=14, ha='center', color='#424242')
    
    # New model architecture
    # Input
    input_box = FancyBboxPatch((1, 9), 8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=new_color, 
                               edgecolor='#2E7D32', linewidth=2)
    ax2.add_patch(input_box)
    ax2.text(5, 9.5, 'Input (60, 3)', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    # Conv layers
    conv1_box = FancyBboxPatch((1, 7.5), 8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=new_color, 
                               edgecolor='#2E7D32', linewidth=2)
    ax2.add_patch(conv1_box)
    ax2.text(5, 8, 'Conv1D: 16 filters (160 params)', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    conv2_box = FancyBboxPatch((1, 6), 8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=new_color, 
                               edgecolor='#2E7D32', linewidth=2)
    ax2.add_patch(conv2_box)
    ax2.text(5, 6.5, 'Conv1D: 32 filters (1,568 params)', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    # Global pooling
    pool_box = FancyBboxPatch((1, 4.5), 8, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=new_color, 
                              edgecolor='#2E7D32', linewidth=2)
    ax2.add_patch(pool_box)
    ax2.text(5, 5, 'GlobalAvgPool1D', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    # Dense layer
    dense_box = FancyBboxPatch((1, 3), 8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=new_color, 
                               edgecolor='#2E7D32', linewidth=2)
    ax2.add_patch(dense_box)
    ax2.text(5, 3.5, 'Dense: 32 units (1,056 params)', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    # Output
    output_box = FancyBboxPatch((1, 1.5), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=new_color, 
                                edgecolor='#2E7D32', linewidth=2)
    ax2.add_patch(output_box)
    ax2.text(5, 2, '3 Output Heads (297 params)', ha='center', va='center', 
             fontsize=12, fontweight='bold', color=text_color)
    
    # Benefits box
    benefits_box = FancyBboxPatch((0.2, 0.2), 9.6, 1.2, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#E8F5E8', 
                                  edgecolor='#2E7D32', linewidth=2)
    ax2.add_patch(benefits_box)
    ax2.text(5, 0.8, 'BENEFITS:', ha='center', va='center', 
             fontsize=12, fontweight='bold', color='#2E7D32')
    ax2.text(5, 0.5, '• 42 parameters per training sample (healthy ratio)', ha='center', va='center', 
             fontsize=10, color='#2E7D32')
    ax2.text(5, 0.2, '• Learns patterns, generalizes well', ha='center', va='center', 
             fontsize=10, color='#2E7D32')
    
    # Comparison metrics in the middle
    comparison_text = """
COMPARISON METRICS

Parameters: 177,000 → 3,180 (55x smaller!)
Accuracy: 26% → 70.7% (2.7x better!)
Training Time: Hours → Minutes
Memory Usage: High → Low
Overfitting: Severe → Controlled
Generalization: Poor → Good

THE GOLDILOCKS ZONE:
• Too few parameters: Can't learn
• Too many parameters: Overfits  
• Just right: Learns without memorizing
    """
    
    # Add comparison text in the middle
    fig.text(0.5, 0.5, comparison_text, ha='center', va='center', 
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
             transform=fig.transFigure)
    
    plt.tight_layout()
    plt.savefig('/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/presentation_visuals/model_complexity_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    create_model_comparison()
