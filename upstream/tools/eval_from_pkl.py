"""
Evaluate TV3S predictions using the same dataset.evaluate() method as test.py.
Fast because it uses get_gt_seg_maps() directly without loading through slow pipeline.
"""

import argparse
import os
import pickle
import sys
import mmcv

sys.path.insert(0, '/workspace/TV3S')

# Register custom datasets
import utils.datasets.vspw

from mmseg.datasets import build_dataset
from mmseg.core.evaluation import eval_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate TV3S predictions')
    parser.add_argument('predictions', help='Path to predictions.pkl file')
    parser.add_argument('config', help='Path to config file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    print("="*70)
    print("TV3S EVALUATION (using dataset.evaluate())")
    print("="*70)
    
    # Load config
    print(f"\nLoading config from: {args.config}")
    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    
    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions, 'rb') as f:
        predictions = pickle.load(f)
    print(f"✓ Loaded {len(predictions)} predictions")
    
    # Build dataset (just for get_gt_seg_maps() access)
    print(f"\nBuilding dataset...")
    dataset = build_dataset(cfg.data.test)
    print(f"✓ Dataset built with {len(dataset)} samples")
    
    # Get ground truth maps using the same method as test.py
    print(f"\nDataset ready (get_gt_seg_maps will be called by evaluate())")
    
    # Evaluate using dataset.evaluate() - same as test.py does
    print(f"\nEvaluating predictions...")
    eval_results = dataset.evaluate(predictions, metric=['mIoU'])
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(eval_results)
    print("="*70)

if __name__ == '__main__':
    main()
