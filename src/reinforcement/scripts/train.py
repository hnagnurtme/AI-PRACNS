#!/usr/bin/env python
"""
Training script for SAGIN RL routing agent.

Usage:
    python scripts/train.py --config configs/dynamic_config.yaml --episodes 2000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
from datetime import datetime
from training.train import DynamicTrainingManager


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train SAGIN RL Routing Agent')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/dynamic_config.yaml',
        help='Path to configuration file (default: configs/dynamic_config.yaml)'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of training episodes (overrides config)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name for logging'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Device to use for training'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Override with command line arguments
    if args.episodes:
        config['training']['num_episodes'] = args.episodes

    if args.checkpoint:
        config['checkpoint_path'] = args.checkpoint

    if args.device != 'auto':
        config['device'] = args.device

    # Create experiment name
    experiment_name = args.name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config['experiment_name'] = experiment_name

    # Print configuration summary
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Experiment Name: {experiment_name}")
    print(f"Config File: {args.config}")
    print(f"Episodes: {config['training']['num_episodes']}")
    print(f"Max Hops: {config['training']['max_hops']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Device: {args.device}")

    if 'dynamics' in config:
        print("\nDynamics:")
        print(f"  Weather: {config['dynamics'].get('weather', {}).get('enabled', False)}")
        print(f"  Traffic: {config['dynamics'].get('traffic', {}).get('enabled', False)}")
        print(f"  Failures: {config['dynamics'].get('failures', {}).get('enabled', False)}")
        print(f"  Mobility: {config['dynamics'].get('mobility', {}).get('enabled', False)}")

    print("="*70 + "\n")

    # Confirm before starting
    response = input("Start training? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Training cancelled.")
        return

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = DynamicTrainingManager(config)

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        trainer.agent.q_network.load_state_dict(torch.load(args.checkpoint))
        print("Checkpoint loaded successfully!")

    # Start training
    print("\nStarting training...")
    print("="*70 + "\n")

    try:
        trainer.train(num_episodes=config['training']['num_episodes'])
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        response = input("Save current checkpoint? [Y/n]: ").strip().lower()
        if not response or response == 'y':
            save_path = f"checkpoints/models/interrupted_{experiment_name}.pth"
            trainer.agent.save_checkpoint(save_path)
            print(f"Checkpoint saved to: {save_path}")

    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()

        response = input("\nSave current checkpoint anyway? [Y/n]: ").strip().lower()
        if not response or response == 'y':
            save_path = f"checkpoints/models/error_{experiment_name}.pth"
            trainer.agent.save_checkpoint(save_path)
            print(f"Checkpoint saved to: {save_path}")
        return 1

    return 0


if __name__ == "__main__":
    import torch  # Import here to avoid circular dependency
    exit(main())
