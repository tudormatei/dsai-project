"""
Visualize masking on real sensor data from the dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mask_reconstruct_pretraining import SensorDataDataset


def visualize_real_data_masking(csv_path: str, user_id: int = 1, activity: str = 'Walking', 
                                window_size: int = 128, mask_ratio: float = 0.15):
    """
    Visualize masking on real sensor data.
    
    Args:
        csv_path: Path to the dataset CSV
        user_id: User ID to visualize
        activity: Activity to visualize
        window_size: Size of time windows
        mask_ratio: Ratio of timesteps to mask
    """
    print(f"Loading data for User {user_id}, Activity: {activity}")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter for specific user and activity
    user_data = df[(df['user'] == user_id) & (df['activity'] == activity)].copy()
    
    if len(user_data) == 0:
        print(f"No data found for User {user_id} with activity {activity}")
        print("Available users:", df['user'].unique())
        print("Available activities:", df['activity'].unique())
        return
    
    print(f"Found {len(user_data)} samples for User {user_id}, Activity: {activity}")
    
    # Extract sensor data
    sensor_data = user_data[['x-axis', 'y-axis', 'z-axis']].values
    
    # Normalize the data
    scaler = StandardScaler()
    sensor_data = scaler.fit_transform(sensor_data)
    
    # Take a subset for visualization (first 1000 samples)
    sensor_data = sensor_data[:1000]
    
    print(f"Using {len(sensor_data)} samples for visualization")
    
    # Create dataset
    dataset = SensorDataDataset(sensor_data, window_size=window_size, mask_ratio=mask_ratio)
    
    # Get a sample window
    sample = dataset[0]
    
    # Create the visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot each axis
    axes[0].plot(sample['target'][:, 0].numpy(), label='Original X', alpha=0.8, linewidth=1.5)
    axes[0].plot(sample['input'][:, 0].numpy(), label='Masked X', alpha=0.8, linewidth=1.5)
    axes[0].set_title(f'X-axis - User {user_id}, {activity}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Acceleration (normalized)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(sample['target'][:, 1].numpy(), label='Original Y', alpha=0.8, linewidth=1.5)
    axes[1].plot(sample['input'][:, 1].numpy(), label='Masked Y', alpha=0.8, linewidth=1.5)
    axes[1].set_title(f'Y-axis - User {user_id}, {activity}', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Acceleration (normalized)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(sample['target'][:, 2].numpy(), label='Original Z', alpha=0.8, linewidth=1.5)
    axes[2].plot(sample['input'][:, 2].numpy(), label='Masked Z', alpha=0.8, linewidth=1.5)
    axes[2].set_title(f'Z-axis - User {user_id}, {activity}', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Acceleration (normalized)')
    axes[2].set_xlabel('Time Steps')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = f'/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining/real_data_masking_user{user_id}_{activity}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    # Print some statistics
    mask = sample['mask'].numpy()
    print(f"\nMasking Statistics:")
    print(f"Total timesteps: {len(mask)}")
    print(f"Masked timesteps: {mask.sum()}")
    print(f"Mask ratio: {mask.sum() / len(mask):.3f}")
    print(f"Original data range: [{sample['target'].min():.3f}, {sample['target'].max():.3f}]")
    print(f"Masked data range: [{sample['input'].min():.3f}, {sample['input'].max():.3f}]")


def compare_activities(csv_path: str, user_id: int = 1, activities: list = None):
    """
    Compare masking across different activities for the same user.
    
    Args:
        csv_path: Path to the dataset CSV
        user_id: User ID to visualize
        activities: List of activities to compare
    """
    if activities is None:
        activities = ['Walking', 'Jogging', 'Sitting', 'Standing']
    
    print(f"Comparing activities for User {user_id}")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter for specific user
    user_data = df[df['user'] == user_id].copy()
    
    if len(user_data) == 0:
        print(f"No data found for User {user_id}")
        return
    
    # Create subplots
    fig, axes = plt.subplots(len(activities), 3, figsize=(18, 4*len(activities)))
    if len(activities) == 1:
        axes = axes.reshape(1, -1)
    
    for i, activity in enumerate(activities):
        activity_data = user_data[user_data['activity'] == activity]
        
        if len(activity_data) == 0:
            print(f"No data found for {activity}")
            continue
        
        # Extract sensor data
        sensor_data = activity_data[['x-axis', 'y-axis', 'z-axis']].values
        
        # Normalize
        scaler = StandardScaler()
        sensor_data = scaler.fit_transform(sensor_data)
        
        # Take subset
        sensor_data = sensor_data[:500]  # Smaller subset for comparison
        
        # Create dataset
        dataset = SensorDataDataset(sensor_data, window_size=64, mask_ratio=0.15)
        
        if len(dataset) == 0:
            print(f"No windows created for {activity}")
            continue
        
        # Get sample
        sample = dataset[0]
        
        # Plot each axis
        for j, axis in enumerate(['X', 'Y', 'Z']):
            axes[i, j].plot(sample['target'][:, j].numpy(), label=f'Original {axis}', alpha=0.8, linewidth=1)
            axes[i, j].plot(sample['input'][:, j].numpy(), label=f'Masked {axis}', alpha=0.8, linewidth=1)
            axes[i, j].set_title(f'{activity} - {axis}-axis', fontsize=12, fontweight='bold')
            axes[i, j].set_ylabel('Acceleration')
            axes[i, j].legend()
            axes[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = f'/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining/activity_comparison_user{user_id}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Activity comparison saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to run visualizations."""
    csv_path = '/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/dataset.csv'
    
    print("=" * 60)
    print("REAL SENSOR DATA MASKING VISUALIZATION")
    print("=" * 60)
    
    # Visualize specific user and activity
    print("\n1. Visualizing User 1, Walking activity...")
    visualize_real_data_masking(csv_path, user_id=1, activity='Walking')
    
    # Compare different activities
    print("\n2. Comparing different activities for User 1...")
    compare_activities(csv_path, user_id=1, activities=['Walking', 'Jogging', 'Sitting'])
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

