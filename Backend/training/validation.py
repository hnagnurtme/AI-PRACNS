"""
Phase 1 Enhancement: Basic Validation Framework
Compare RL vs Dijkstra performance
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


class RLValidator:
    """
    Comprehensive validation framework để so sánh RL vs Dijkstra
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.results = {}
    
    def validate_against_dijkstra(
        self,
        agent,
        nodes: List[Dict],
        terminals: List[Dict],
        num_tests: int = 100,
        deterministic: bool = True
    ) -> Dict:
        """
        Compare RL vs Dijkstra trên nhiều test cases
        
        Args:
            agent: RL agent instance
            nodes: List of nodes
            terminals: List of terminals
            num_tests: Number of test cases
            deterministic: Use deterministic action selection for RL
            
        Returns:
            Dictionary với comparison results
        """
        logger.info(f"Starting validation: {num_tests} test cases")
        
        results = {
            'rl_success': 0,
            'dijkstra_success': 0,
            'both_success': 0,
            'rl_better': 0,
            'dijkstra_better': 0,
            'equal': 0,
            'rl_hops': [],
            'dijkstra_hops': [],
            'rl_latency': [],
            'dijkstra_latency': [],
            'rl_distance': [],
            'dijkstra_distance': [],
            'rl_qos_violations': 0,
            'dijkstra_qos_violations': 0,
        }
        
        # Import routing functions
        from api.routing_bp import calculate_path_dijkstra
        from services.rl_routing_service import get_rl_routing_service
        from config import Config
        
        rl_service = get_rl_routing_service(Config.get_yaml_config())
        
        for i in range(num_tests):
            if len(terminals) < 2:
                logger.warning("Not enough terminals for validation")
                break
            
            # Random terminal pair
            source, dest = random.sample(terminals, 2)
            
            try:
                # RL path
                rl_path = rl_service.calculate_path_rl(
                    source_terminal=source,
                    dest_terminal=dest,
                    nodes=nodes
                )
                
                # Dijkstra path
                dijkstra_path = calculate_path_dijkstra(
                    source_terminal=source,
                    dest_terminal=dest,
                    nodes=nodes
                )
                
                # Compare results
                rl_success = rl_path.get('success', False)
                dij_success = dijkstra_path.get('success', False)
                
                if rl_success:
                    results['rl_success'] += 1
                    results['rl_hops'].append(rl_path.get('hops', 0))
                    results['rl_latency'].append(rl_path.get('estimatedLatency', 0))
                    results['rl_distance'].append(rl_path.get('totalDistance', 0))
                
                if dij_success:
                    results['dijkstra_success'] += 1
                    results['dijkstra_hops'].append(dijkstra_path.get('hops', 0))
                    results['dijkstra_latency'].append(dijkstra_path.get('estimatedLatency', 0))
                    results['dijkstra_distance'].append(dijkstra_path.get('totalDistance', 0))
                
                if rl_success and dij_success:
                    results['both_success'] += 1
                    
                    # Compare metrics
                    rl_hops = rl_path.get('hops', 0)
                    dij_hops = dijkstra_path.get('hops', 0)
                    
                    if rl_hops < dij_hops:
                        results['rl_better'] += 1
                    elif dij_hops < rl_hops:
                        results['dijkstra_better'] += 1
                    else:
                        results['equal'] += 1
                
                # QoS violations (if QoS provided)
                # This would need QoS requirements to be passed in
                
            except Exception as e:
                logger.warning(f"Validation test {i+1} failed: {e}")
                continue
        
        # Calculate statistics
        stats = self._calculate_statistics(results)
        results.update(stats)
        
        logger.info(f"Validation completed:")
        logger.info(f"  RL Success: {results['rl_success']}/{num_tests} ({results['rl_success_rate']:.1f}%)")
        logger.info(f"  Dijkstra Success: {results['dijkstra_success']}/{num_tests} ({results['dijkstra_success_rate']:.1f}%)")
        logger.info(f"  RL Better: {results['rl_better']}/{results['both_success']} ({results['rl_better_rate']:.1f}%)")
        logger.info(f"  Dijkstra Better: {results['dijkstra_better']}/{results['both_success']} ({results['dijkstra_better_rate']:.1f}%)")
        
        return results
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """Calculate statistics từ results"""
        stats = {}
        
        # Success rates
        total_tests = max(
            results['rl_success'] + (results.get('rl_failures', 0) if 'rl_failures' in results else 0),
            results['dijkstra_success'] + (results.get('dijkstra_failures', 0) if 'dijkstra_failures' in results else 0),
            1
        )
        
        stats['rl_success_rate'] = (results['rl_success'] / total_tests * 100) if total_tests > 0 else 0
        stats['dijkstra_success_rate'] = (results['dijkstra_success'] / total_tests * 100) if total_tests > 0 else 0
        
        # Comparison rates
        both_success = results['both_success']
        if both_success > 0:
            stats['rl_better_rate'] = results['rl_better'] / both_success * 100
            stats['dijkstra_better_rate'] = results['dijkstra_better'] / both_success * 100
            stats['equal_rate'] = results['equal'] / both_success * 100
        else:
            stats['rl_better_rate'] = 0
            stats['dijkstra_better_rate'] = 0
            stats['equal_rate'] = 0
        
        # Average metrics
        if results['rl_hops']:
            stats['rl_avg_hops'] = np.mean(results['rl_hops'])
            stats['rl_std_hops'] = np.std(results['rl_hops'])
        else:
            stats['rl_avg_hops'] = 0
            stats['rl_std_hops'] = 0
        
        if results['dijkstra_hops']:
            stats['dijkstra_avg_hops'] = np.mean(results['dijkstra_hops'])
            stats['dijkstra_std_hops'] = np.std(results['dijkstra_hops'])
        else:
            stats['dijkstra_avg_hops'] = 0
            stats['dijkstra_std_hops'] = 0
        
        if results['rl_latency']:
            stats['rl_avg_latency'] = np.mean(results['rl_latency'])
            stats['rl_std_latency'] = np.std(results['rl_latency'])
        else:
            stats['rl_avg_latency'] = 0
            stats['rl_std_latency'] = 0
        
        if results['dijkstra_latency']:
            stats['dijkstra_avg_latency'] = np.mean(results['dijkstra_latency'])
            stats['dijkstra_std_latency'] = np.std(results['dijkstra_latency'])
        else:
            stats['dijkstra_avg_latency'] = 0
            stats['dijkstra_std_latency'] = 0
        
        # Hops ratio (RL / Dijkstra)
        if stats['dijkstra_avg_hops'] > 0:
            stats['hops_ratio'] = stats['rl_avg_hops'] / stats['dijkstra_avg_hops']
        else:
            stats['hops_ratio'] = float('inf')
        
        # Latency ratio (RL / Dijkstra)
        if stats['dijkstra_avg_latency'] > 0:
            stats['latency_ratio'] = stats['rl_avg_latency'] / stats['dijkstra_avg_latency']
        else:
            stats['latency_ratio'] = float('inf')
        
        return stats
    
    def save_results(self, results: Dict, filepath: str):
        """Save validation results to file"""
        import json
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Validation results saved to {filepath}")

