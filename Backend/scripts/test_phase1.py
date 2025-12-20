#!/usr/bin/env python3
"""
Phase 1 Testing Script
Test enhanced state representation and Dijkstra-aligned rewards
"""
import sys
import os
from pathlib import Path

# Add Backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import logging
import numpy as np
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config import Config
from models.database import db
from environment.state_builder import RoutingStateBuilder
from environment.routing_env import RoutingEnvironment
from training.validation import RLValidator


def test_state_builder():
    """Test enhanced state builder với Dijkstra-like features"""
    logger.info("=" * 60)
    logger.info("TEST 1: State Builder Enhancement")
    logger.info("=" * 60)
    
    try:
        # Load config
        config = Config.get_yaml_config()
        
        # Initialize state builder
        state_builder = RoutingStateBuilder(config)
        
        logger.info(f"State dimension: {state_builder.state_dimension}")
        logger.info(f"Node feature dim: {state_builder.node_feature_dim}")
        logger.info(f"Max nodes: {state_builder.max_nodes}")
        logger.info(f"Include Dijkstra features: {state_builder.include_dijkstra_features}")
        
        # Get test data
        db.connect()
        nodes_collection = db.get_collection('nodes')
        terminals_collection = db.get_collection('terminals')
        
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}).limit(10))
        terminals = list(terminals_collection.find({}, {'_id': 0}).limit(2))
        
        if len(nodes) < 2 or len(terminals) < 2:
            logger.error("Not enough test data")
            return False
        
        source_terminal = terminals[0]
        dest_terminal = terminals[1]
        
        # Test state building
        state = state_builder.build_state(
            nodes=nodes,
            source_terminal=source_terminal,
            dest_terminal=dest_terminal,
            current_node=nodes[0] if nodes else None,
            visited_nodes=[]
        )
        
        logger.info(f"✅ State built successfully")
        logger.info(f"   State shape: {state.shape}")
        logger.info(f"   State dimension: {len(state)}")
        logger.info(f"   Expected dimension: {state_builder.state_dimension}")
        logger.info(f"   State range: [{state.min():.3f}, {state.max():.3f}]")
        logger.info(f"   State mean: {state.mean():.3f}, std: {state.std():.3f}")
        
        # Verify dimension matches
        if len(state) == state_builder.state_dimension:
            logger.info("✅ State dimension matches expected")
        else:
            logger.error(f"❌ State dimension mismatch: {len(state)} != {state_builder.state_dimension}")
            return False
        
        # Test Dijkstra edge weight estimation
        if len(nodes) >= 2:
            current_node = nodes[0]
            next_node = nodes[1]
            current_pos = current_node.get('position')
            next_pos = next_node.get('position')
            
            if current_pos and next_pos:
                weight = state_builder._estimate_dijkstra_edge_weight(
                    current_node, next_node, current_pos, next_pos
                )
                logger.info(f"✅ Dijkstra edge weight estimated: {weight:.2f} km")
        
        # Close connection if method exists
        if hasattr(db, 'disconnect'):
            db.disconnect()
        elif hasattr(db, 'close'):
            db.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ State builder test failed: {e}", exc_info=True)
        # Try to close connection
        try:
            if hasattr(db, 'disconnect'):
                db.disconnect()
            elif hasattr(db, 'close'):
                db.close()
        except:
            pass
        return False


def test_reward_function():
    """Test Dijkstra-aligned reward function"""
    logger.info("=" * 60)
    logger.info("TEST 2: Dijkstra-Aligned Reward Function")
    logger.info("=" * 60)
    
    try:
        # Load config
        config = Config.get_yaml_config()
        
        # Get test data
        db.connect()
        nodes_collection = db.get_collection('nodes')
        terminals_collection = db.get_collection('terminals')
        
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}).limit(10))
        terminals = list(terminals_collection.find({}, {'_id': 0}).limit(2))
        
        if len(nodes) < 2 or len(terminals) < 2:
            logger.error("Not enough test data")
            return False
        
        # Initialize environment
        env = RoutingEnvironment(
            nodes=nodes,
            terminals=terminals,
            config=config,
            max_steps=10
        )
        
        logger.info(f"Use Dijkstra-aligned rewards: {env.use_dijkstra_aligned_rewards}")
        logger.info(f"Drop threshold: {env.drop_threshold}%")
        logger.info(f"Penalty threshold: {env.penalty_threshold}%")
        logger.info(f"Penalty multiplier: {env.penalty_multiplier}x")
        
        # Reset environment
        state, info = env.reset(
            options={
                'source_terminal_id': terminals[0].get('terminalId'),
                'dest_terminal_id': terminals[1].get('terminalId')
            }
        )
        
        logger.info(f"✅ Environment reset successfully")
        logger.info(f"   Current node: {env.current_node.get('nodeId') if env.current_node else 'None'}")
        logger.info(f"   Path length: {len(env.path)}")
        
        # Test reward calculation
        if env.current_node and len(nodes) > 1:
            current_node = env.current_node
            next_node = nodes[1] if nodes[1].get('nodeId') != current_node.get('nodeId') else nodes[2] if len(nodes) > 2 else None
            
            if next_node:
                current_pos = current_node.get('position')
                next_pos = next_node.get('position')
                dest_pos = terminals[1].get('position')
                
                if current_pos and next_pos and dest_pos:
                    distance = env._calculate_distance(current_pos, next_pos)
                    reward = env._calculate_dijkstra_aligned_reward(
                        current_node, next_node, distance, dest_pos
                    )
                    
                    logger.info(f"✅ Dijkstra-aligned reward calculated: {reward:.2f}")
                    logger.info(f"   Distance: {distance/1000:.2f} km")
                    logger.info(f"   Next node utilization: {max(next_node.get('cpu', {}).get('utilization', 0), next_node.get('memory', {}).get('utilization', 0), next_node.get('bandwidth', {}).get('utilization', 0)):.1f}%")
        
        # Close connection if method exists
        if hasattr(db, 'disconnect'):
            db.disconnect()
        elif hasattr(db, 'close'):
            db.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Reward function test failed: {e}", exc_info=True)
        # Try to close connection
        try:
            if hasattr(db, 'disconnect'):
                db.disconnect()
            elif hasattr(db, 'close'):
                db.close()
        except:
            pass
        return False


def test_validation_framework():
    """Test validation framework"""
    logger.info("=" * 60)
    logger.info("TEST 3: Validation Framework")
    logger.info("=" * 60)
    
    try:
        # Load config
        config = Config.get_yaml_config()
        
        # Get test data
        db.connect()
        nodes_collection = db.get_collection('nodes')
        terminals_collection = db.get_collection('terminals')
        
        nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
        terminals = list(terminals_collection.find({}, {'_id': 0}))
        
        if len(nodes) < 2 or len(terminals) < 2:
            logger.error("Not enough test data")
            return False
        
        logger.info(f"Loaded {len(nodes)} nodes and {len(terminals)} terminals")
        
        # Initialize validator
        validator = RLValidator(config)
        logger.info("✅ Validator initialized")
        
        # Note: Full validation requires trained model, so we'll just test the framework
        logger.info("✅ Validation framework ready")
        logger.info("   (Full validation requires trained RL model)")
        
        # Close connection if method exists
        if hasattr(db, 'disconnect'):
            db.disconnect()
        elif hasattr(db, 'close'):
            db.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation framework test failed: {e}", exc_info=True)
        # Try to close connection
        try:
            if hasattr(db, 'disconnect'):
                db.disconnect()
            elif hasattr(db, 'close'):
                db.close()
        except:
            pass
        return False


def main():
    """Run all Phase 1 tests"""
    logger.info("=" * 60)
    logger.info("PHASE 1 TESTING: Foundation Enhancements")
    logger.info("=" * 60)
    
    results = {
        'state_builder': False,
        'reward_function': False,
        'validation_framework': False
    }
    
    # Test 1: State Builder
    results['state_builder'] = test_state_builder()
    
    # Test 2: Reward Function
    results['reward_function'] = test_reward_function()
    
    # Test 3: Validation Framework
    results['validation_framework'] = test_validation_framework()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("=" * 60)
        logger.info("✅ ALL TESTS PASSED - Phase 1 is working correctly!")
        logger.info("=" * 60)
        return 0
    else:
        logger.info("=" * 60)
        logger.error("❌ SOME TESTS FAILED - Please check errors above")
        logger.info("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())

