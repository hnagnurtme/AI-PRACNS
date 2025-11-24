import yaml
import argparse
from training.train import DynamicTrainingManager

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Dynamic SAGIN RL Training')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'demo'], 
                       default='train', help='Run mode')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.mode == 'train':
        trainer = DynamicTrainingManager(config)
        trainer.train(num_episodes=config['training']['num_episodes'])
    elif args.mode == 'eval':
        from training.evaluator import Evaluator
        evaluator = Evaluator(config)
        evaluator.compare_algorithms()
    elif args.mode == 'demo':
        try:
            try:
                from simulation.scenarios.demo import run_demo
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Ensure 'simulation/scenarios/demo.py' exists and is in the correct path.")
        except ImportError:
            raise ImportError("Module 'simulation.scenarios.demo' not found. Verify the module path or ensure it exists.")
        run_demo(config)

if __name__ == "__main__":
    main()