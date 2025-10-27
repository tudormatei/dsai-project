"""
Create multi-dimensional radar chart comparing splitting strategies
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

def create_radar_chart():
    """Create a radar chart comparing 6 splitting strategies across 3 dimensions"""
    
    # Data for the radar chart
    categories = ['User\nGeneralization', 'Concept\nLearning', 'Deployment\nReality']
    
    # Strategy data (normalized 0-1 scale)
    strategies = {
        'LOUO': [1.0, 0.25, 1.0],  # High generalization, low learning, high reality
        'LOUO + Aug': [1.0, 0.35, 1.0],  # High generalization, low learning, high reality
        'Random Split': [0.25, 0.8, 0.25],  # Low generalization, high learning, low reality
        'Random + Aug': [0.25, 1.0, 0.25],  # Low generalization, very high learning, low reality
        'Advanced CV': [0.75, 0.6, 0.75],  # Good generalization, good learning, good reality
        'Advanced + Aug': [0.75, 1.0, 0.75]  # Good generalization, very high learning, good reality
    }
    
    # Colors for each strategy
    colors = {
        'LOUO': '#FFCDD2',  # Light red
        'LOUO + Aug': '#FFAB91',  # Light orange
        'Random Split': '#E1F5FE',  # Light blue
        'Random + Aug': '#B3E5FC',  # Medium blue
        'Advanced CV': '#E8F5E8',  # Light green
        'Advanced + Aug': '#C8E6C9'  # Medium green
    }
    
    # Line colors
    line_colors = {
        'LOUO': '#D32F2F',
        'LOUO + Aug': '#FF5722',
        'Random Split': '#1976D2',
        'Random + Aug': '#2196F3',
        'Advanced CV': '#388E3C',
        'Advanced + Aug': '#4CAF50'
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Set up the plot
    ax.set_theta_offset(np.pi / 2)  # Rotate plot
    ax.set_theta_direction(-1)  # Reverse direction
    
    # Draw one tick per variable + add names
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    
    # Draw ylabels
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot data for each strategy
    for strategy, values in strategies.items():
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=strategy, color=line_colors[strategy])
        ax.fill(angles, values, alpha=0.25, color=colors[strategy])
    
    # Add title
    plt.title('Multi-Dimensional Analysis of Data Splitting Strategies\n', 
              size=16, fontweight='bold', pad=20)
    
    # Add legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
                      fontsize=10, frameon=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add performance metrics as text
    performance_text = """
Performance Metrics:
• LOUO: 21.7% accuracy
• LOUO + Aug: 26.1% accuracy  
• Random Split: 49.3% accuracy
• Random + Aug: 70.7% accuracy ⭐
• Advanced CV: 52.7% accuracy
• Advanced + Aug: 68.5% accuracy
    """
    
    ax.text(0.02, 0.98, performance_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add dimension explanations
    dimension_text = """
Dimension Explanations:
• User Generalization: How well it works for new users
• Concept Learning: How well it learns patterns  
• Deployment Reality: How realistic for real-world use
    """
    
    ax.text(0.02, 0.15, dimension_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/presentation_visuals/splitting_strategies_radar_chart.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    create_radar_chart()
