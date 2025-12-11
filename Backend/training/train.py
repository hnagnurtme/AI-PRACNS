"""
Optimized Training script for DuelingDQN Routing Agent
Usage: python -m training.train [--episodes 2000] [--config custom_config.yaml]
"""
import sys
import os
import logging
import argparse
import time
from pathlib import Path

# Add Backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from config import Config
from training.trainer import RoutingTrainer
from training.enhanced_trainer import EnhancedRoutingTrainer
from models.database import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DuelingDQN Routing Agent')
    parser.add_argument('--episodes', type=int, default=None, 
                       help='Number of training episodes')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing model')
    
    return parser.parse_args()


def main():
    """Optimized main training function"""
    args = parse_arguments()
    
    try:
        # Initialize configuration
        if args.config:
            config = Config.load_from_file(args.config)
        else:
            config = Config.get_yaml_config()
        
        logger.info("Starting optimized RL routing training...")
        
        # Initialize database connection
        db.connect()
        logger.info("Database connected successfully")
        
        # Initialize trainer (enhanced or standard)
        use_enhanced = config.get('training', {}).get('use_enhanced_trainer', False)
        if use_enhanced:
            logger.info("Using Enhanced Trainer with Curriculum Learning, Imitation Learning, and Multi-objective Optimization")
            trainer = EnhancedRoutingTrainer(config)
        else:
            logger.info("Using Standard Trainer")
            trainer = RoutingTrainer(config)
        
        if args.eval_only:
            # Evaluation mode
            logger.info("Running evaluation only...")
            _run_evaluation(trainer)
        else:
            # Training mode
            logger.info("Starting training from database...")
            
            start_time = time.time()
            agent = trainer.train_from_database(num_episodes=args.episodes)
            
            training_time = time.time() - start_time
            logger.info(f"Training completed successfully in {training_time:.2f} seconds!")
            
            # Run final evaluation
            _run_final_evaluation(trainer, agent)
            
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Cleanup
        if 'db' in locals():
            db.disconnect()
            logger.info("Database disconnected")


def _run_evaluation(trainer: RoutingTrainer):
    """Run comprehensive evaluation"""
    try:
        from models.database import db
        
        # Load test data
        nodes_collection = db.get_collection('nodes')
        terminals_collection = db.get_collection('terminals')
        
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        terminals = list(terminals_collection.find({}, {'_id': 0}))
        
        if len(nodes) == 0 or len(terminals) < 2:
            logger.error("Insufficient data for evaluation")
            return
        
        # Find best model
        best_model_path = Path('./models/best_models/best_model.pt')
        if not best_model_path.exists():
            logger.error("No best model found for evaluation")
            return
        
        # Run evaluation
        logger.info("Running comprehensive evaluation...")
        metrics = trainer.load_and_evaluate(
            model_path=str(best_model_path),
            nodes=nodes,
            terminals=terminals,
            num_episodes=50
        )
        
        logger.info("Evaluation Results:")
        logger.info(f"  Mean Reward: {metrics['mean_reward']:.2f}")
        logger.info(f"  Success Rate: {metrics['success_rate']:.2f}")
        logger.info(f"  Average Hops: {metrics['mean_hops']:.1f}")
        logger.info(f"  Average Latency: {metrics['mean_latency']:.2f}ms")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)


def _run_final_evaluation(trainer: RoutingTrainer, agent):
    """Run final evaluation after training"""
    logger.info("Running final evaluation...")
    
    try:
        # Simple evaluation using the trained agent
        from models.database import db
        
        nodes_collection = db.get_collection('nodes')
        terminals_collection = db.get_collection('terminals')
        
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        terminals = list(terminals_collection.find({}, {'_id': 0}))
        
        if len(nodes) > 0 and len(terminals) >= 2:
            # Create environment for evaluation
            from environment.routing_env import RoutingEnvironment
            env = RoutingEnvironment(
                nodes=nodes,
                terminals=terminals,
                config=trainer.config
            )
            
            # Run evaluation
            eval_reward, eval_metrics = trainer.evaluate_advanced(
                agent, env, num_episodes=20
            )
            
            logger.info("Final Evaluation Results:")
            logger.info(f"  Mean Reward: {eval_reward:.2f}")
            logger.info(f"  Success Rate: {eval_metrics['success_rate']:.2f}")
            logger.info(f"  Average Hops: {eval_metrics['mean_hops']:.1f}")
            
        else:
            logger.warning("Insufficient data for final evaluation")
            
    except Exception as e:
        logger.warning(f"Final evaluation skipped: {e}")


if __name__ == '__main__':
    main()