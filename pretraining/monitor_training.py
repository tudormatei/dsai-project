"""
Monitor the pretraining progress.
"""

import os
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def check_training_status():
    """Check if training is still running."""
    try:
        result = subprocess.run(['pgrep', '-f', 'improved_pretraining.py'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False


def parse_log_file(log_path):
    """Parse the training log to extract loss values."""
    if not os.path.exists(log_path):
        return None, None
    
    train_losses = []
    val_losses = []
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if 'Train Loss:' in line and 'Val Loss:' in line:
                # Extract loss values
                parts = line.split('Train Loss:')[1].split(',')
                train_loss = float(parts[0].strip())
                val_loss = float(parts[1].split('Val Loss:')[1].strip())
                train_losses.append(train_loss)
                val_losses.append(val_loss)
    
    except Exception as e:
        print(f"Error parsing log: {e}")
    
    return train_losses, val_losses


def plot_progress(train_losses, val_losses, save_path=None):
    """Plot the training progress."""
    if not train_losses or not val_losses:
        print("No loss data available yet.")
        return
    
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label='Train Loss', alpha=0.8, linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', alpha=0.8, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Pretraining Progress - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def monitor_training():
    """Monitor the training progress."""
    log_path = '/Users/mariarye/Desktop/UNI 2025-2026/DS & AI/Project/dsai-project/pretraining/pretraining.log'
    
    print("Monitoring pretraining progress...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            if check_training_status():
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training is running...")
                
                # Parse and plot progress
                train_losses, val_losses = parse_log_file(log_path)
                if train_losses and val_losses:
                    print(f"Current epoch: {len(train_losses)}")
                    print(f"Latest train loss: {train_losses[-1]:.4f}")
                    print(f"Latest val loss: {val_losses[-1]:.4f}")
                    
                    # Plot progress
                    plot_progress(train_losses, val_losses, 
                                'pretraining_progress.png')
                else:
                    print("Waiting for training to start...")
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training has finished!")
                
                # Final plot
                train_losses, val_losses = parse_log_file(log_path)
                if train_losses and val_losses:
                    print("Final training results:")
                    print(f"Total epochs: {len(train_losses)}")
                    print(f"Final train loss: {train_losses[-1]:.4f}")
                    print(f"Final val loss: {val_losses[-1]:.4f}")
                    
                    plot_progress(train_losses, val_losses, 
                                'final_training_curves.png')
                break
            
            time.sleep(30)  # Check every 30 seconds
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        
        # Show final results
        train_losses, val_losses = parse_log_file(log_path)
        if train_losses and val_losses:
            print("\nFinal training results:")
            print(f"Total epochs completed: {len(train_losses)}")
            print(f"Final train loss: {train_losses[-1]:.4f}")
            print(f"Final val loss: {val_losses[-1]:.4f}")
            
            plot_progress(train_losses, val_losses, 
                        'final_training_curves.png')


if __name__ == "__main__":
    monitor_training()

