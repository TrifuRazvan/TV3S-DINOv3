#!/usr/bin/env python3
"""
Training Log Plotter - Edit the LOG_FILES and OUTPUT_DIR below, then run this script
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# ⭐ EDIT THESE SETTINGS ⭐
# ============================================================================
# Add 1-3 log files to compare (uncomment lines to add them)
LOG_FILES = [
    '/workspace/TV3S/work_dirs/dinov3_vits16_160k_frozen_v2/20251202_001921.log.json',
     '/workspace/TV3S/work_dirs/dinov3_2gpu_4sample.json',
     '/workspace/TV3S/work_dirs/swins_160k_2sample_2gpu.json',
]

# Where to save the plots
OUTPUT_DIR = '/workspace/TV3S/plots'
# ============================================================================

def load_json_logs(log_file):
    """Load training logs from JSON file."""
    with open(log_file, 'r') as f:
        logs = [json.loads(line) for line in f if line.strip()]
    return logs


def plot_logs(log_files, output_dir):
    """Plot training metrics from one or multiple log files.
    
    Args:
        log_files: List of paths to JSON log files
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load all logs
    all_logs = {}
    for log_file in log_files:
        if not Path(log_file).exists():
            print(f"⚠️ File not found: {log_file}")
            continue
        print(f"📂 Loading: {log_file}")
        logs = load_json_logs(log_file)
        all_logs[log_file] = logs
        print(f"   Loaded {len(logs)} entries")
    
    if not all_logs:
        print("❌ No log files found!")
        return
    
    # Extract metrics
    metrics = {}
    for log_file, logs in all_logs.items():
        for log in logs:
            for key, value in log.items():
                if key not in ['timestamp', 'time', 'iter', 'epoch'] and isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = {}
                    if log_file not in metrics[key]:
                        metrics[key][log_file] = {'iter': [], 'value': []}
                    metrics[key][log_file]['iter'].append(log.get('iter', 0))
                    metrics[key][log_file]['value'].append(float(value))
    
    if not metrics:
        print("❌ No metrics found in logs!")
        return
    
    print(f"\n📊 Creating plots for {len(metrics)} metrics...\n")
    
    # Create plots
    for metric, data in metrics.items():
        plt.figure(figsize=(14, 7))
        
        for log_file, values in data.items():
            # Use work_dir name as label
            label = Path(log_file).parent.name
            # Remove timestamp suffix if present
            if '_' in label and label[-15:].replace('_', '').isdigit():
                label = label[:-16]
            
            plt.plot(values['iter'], values['value'], linewidth=2, label=label, alpha=0.8, marker='.')
        
        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel(metric, fontsize=12, fontweight='bold')
        plt.title(f'{metric} vs Iteration', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_file = f'{output_dir}/{metric.replace(".", "_")}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f'✅ {metric:30s} → {output_file}')
        plt.close()
    
    print(f"\n✨ All plots saved to: {output_dir}")


if __name__ == '__main__':
    plot_logs(LOG_FILES, OUTPUT_DIR)
