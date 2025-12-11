"""
Multi-objective Optimization Module
Pareto front cho latency vs reliability vs energy
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveWeights:
    """Weights cho các objectives"""
    latency: float = 0.4
    reliability: float = 0.3
    energy: float = 0.3
    
    def normalize(self):
        """Normalize weights to sum to 1.0"""
        total = self.latency + self.reliability + self.energy
        if total > 0:
            self.latency /= total
            self.reliability /= total
            self.energy /= total


@dataclass
class ParetoSolution:
    """Một solution trên Pareto front"""
    path: List[Dict]
    latency: float
    reliability: float
    energy: float
    objectives: np.ndarray  # [latency, reliability, energy]
    
    def dominates(self, other: 'ParetoSolution') -> bool:
        """Kiểm tra xem solution này có dominate solution khác không"""
        return (self.latency <= other.latency and
                self.reliability >= other.reliability and
                self.energy <= other.energy and
                (self.latency < other.latency or
                 self.reliability > other.reliability or
                 self.energy < other.energy))


class MultiObjectiveOptimizer:
    """
    Multi-objective Optimization với Pareto front
    Tối ưu đồng thời: latency, reliability, energy
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        mo_config = self.config.get('multi_objective', {})
        
        self.enabled = mo_config.get('enabled', True)
        self.use_pareto = mo_config.get('use_pareto', True)
        self.pareto_front_size = mo_config.get('pareto_front_size', 10)
        
        # Objective weights (có thể adaptive)
        self.weights = ObjectiveWeights(
            latency=mo_config.get('latency_weight', 0.4),
            reliability=mo_config.get('reliability_weight', 0.3),
            energy=mo_config.get('energy_weight', 0.3)
        )
        self.weights.normalize()
        
        # Pareto front storage
        self.pareto_front = deque(maxlen=self.pareto_front_size)
        
        # Normalization factors
        self.latency_norm = 1000.0  # ms
        self.reliability_norm = 1.0  # 0-1
        self.energy_norm = 100.0  # arbitrary units
        
        logger.info(f"Multi-objective optimizer initialized (Pareto: {self.use_pareto})")
    
    def compute_objectives(
        self,
        path: List[Dict],
        nodes: List[Dict]
    ) -> Tuple[float, float, float]:
        """
        Compute objectives cho một path:
        - Latency: tổng thời gian delay
        - Reliability: xác suất thành công (dựa trên packet loss, utilization)
        - Energy: tổng năng lượng tiêu thụ
        """
        if len(path) < 2:
            return float('inf'), 0.0, float('inf')
        
        total_latency = 0.0
        total_reliability = 1.0
        total_energy = 0.0
        
        node_map = {n['nodeId']: n for n in nodes}
        
        for i in range(len(path) - 1):
            current = path[i]
            next_item = path[i + 1]
            
            # Get node info
            if current.get('type') == 'node':
                node = node_map.get(current.get('id'))
                if not node:
                    continue
                
                # Latency
                processing_delay = node.get('nodeProcessingDelayMs', 5)
                total_latency += processing_delay
                
                # Reliability (based on packet loss rate and utilization)
                loss_rate = node.get('packetLossRate', 0)
                utilization = node.get('resourceUtilization', 0)
                node_reliability = (1 - loss_rate) * (1 - utilization / 100.0)
                total_reliability *= node_reliability
                
                # Energy (simplified: based on processing delay and utilization)
                # Higher utilization = more energy
                energy_cost = processing_delay * (1 + utilization / 100.0)
                total_energy += energy_cost
            
            # Propagation delay
            if current.get('position') and next_item.get('position'):
                from environment.state_builder import RoutingStateBuilder
                state_builder = RoutingStateBuilder(self.config)
                
                distance = state_builder._calculate_distance(
                    current['position'], next_item['position']
                )
                speed_of_light = 299792458  # m/s
                propagation_delay = (distance / speed_of_light) * 1000  # ms
                total_latency += propagation_delay
                
                # Energy for transmission (simplified)
                transmission_energy = distance / 1000000.0  # Normalized
                total_energy += transmission_energy
        
        return total_latency, total_reliability, total_energy
    
    def compute_scalarized_reward(
        self,
        latency: float,
        reliability: float,
        energy: float
    ) -> float:
        """
        Compute scalarized reward từ multiple objectives
        Sử dụng weighted sum
        """
        # Normalize objectives
        norm_latency = latency / self.latency_norm
        norm_reliability = 1.0 - reliability  # Convert to cost (lower is better)
        norm_energy = energy / self.energy_norm
        
        # Scalarize
        scalarized = (
            self.weights.latency * norm_latency +
            self.weights.reliability * norm_reliability +
            self.weights.energy * norm_energy
        )
        
        # Convert to reward (negative cost)
        reward = -scalarized * 100.0  # Scale
        
        return reward
    
    def update_pareto_front(self, solution: ParetoSolution):
        """Update Pareto front với solution mới"""
        if not self.use_pareto:
            return
        
        # Check if solution is dominated by existing solutions
        is_dominated = False
        to_remove = []
        
        # Convert to list để có thể remove items
        pareto_list = list(self.pareto_front)
        
        for i, existing in enumerate(pareto_list):
            if existing.dominates(solution):
                is_dominated = True
                break
            elif solution.dominates(existing):
                to_remove.append(i)
        
        # Remove dominated solutions (reverse order để không ảnh hưởng indices)
        for i in reversed(to_remove):
            pareto_list.pop(i)
        
        # Add new solution if not dominated
        if not is_dominated:
            pareto_list.append(solution)
            
            # Keep only best solutions if front is full
            if len(pareto_list) > self.pareto_front_size:
                # Sort by hypervolume contribution or diversity
                pareto_list = sorted(pareto_list, key=lambda s: s.latency + s.energy)[:self.pareto_front_size]
            
            # Update deque
            self.pareto_front = deque(pareto_list, maxlen=self.pareto_front_size)
    
    def get_pareto_front(self) -> List[ParetoSolution]:
        """Lấy current Pareto front"""
        return list(self.pareto_front)
    
    def select_from_pareto_front(
        self,
        preference: Optional[Dict] = None
    ) -> Optional[ParetoSolution]:
        """
        Select solution từ Pareto front dựa trên preference
        preference: {'latency': 0.5, 'reliability': 0.3, 'energy': 0.2}
        """
        if len(self.pareto_front) == 0:
            return None
        
        if preference:
            # Weighted selection based on preference
            weights = ObjectiveWeights(
                latency=preference.get('latency', 0.33),
                reliability=preference.get('reliability', 0.33),
                energy=preference.get('energy', 0.33)
            )
            weights.normalize()
            
            best_solution = None
            best_score = float('inf')
            
            for solution in self.pareto_front:
                score = (
                    weights.latency * (solution.latency / self.latency_norm) +
                    weights.reliability * (1 - solution.reliability) +
                    weights.energy * (solution.energy / self.energy_norm)
                )
                
                if score < best_score:
                    best_score = score
                    best_solution = solution
            
            return best_solution
        else:
            # Return solution với best trade-off
            return min(
                self.pareto_front,
                key=lambda s: s.latency + (1 - s.reliability) * 1000 + s.energy
            )
    
    def compute_reward_with_objectives(
        self,
        path: List[Dict],
        nodes: List[Dict],
        use_pareto: bool = True
    ) -> Tuple[float, Dict]:
        """
        Compute reward với multi-objective consideration
        Returns: (reward, objectives_dict)
        """
        latency, reliability, energy = self.compute_objectives(path, nodes)
        
        objectives = {
            'latency': latency,
            'reliability': reliability,
            'energy': energy
        }
        
        if use_pareto and self.use_pareto:
            # Create solution
            solution = ParetoSolution(
                path=path,
                latency=latency,
                reliability=reliability,
                energy=energy,
                objectives=np.array([latency, 1 - reliability, energy])
            )
            
            # Update Pareto front
            self.update_pareto_front(solution)
            
            # Check if solution is on Pareto front (compare by objectives, not by object identity)
            is_pareto = False
            for pareto_solution in self.pareto_front:
                if (np.allclose(pareto_solution.latency, solution.latency, rtol=1e-5) and
                    np.allclose(pareto_solution.reliability, solution.reliability, rtol=1e-5) and
                    np.allclose(pareto_solution.energy, solution.energy, rtol=1e-5)):
                    is_pareto = True
                    break
            
            # Bonus for Pareto solutions
            if is_pareto:
                reward = self.compute_scalarized_reward(latency, reliability, energy)
                reward += 50.0  # Pareto bonus
            else:
                reward = self.compute_scalarized_reward(latency, reliability, energy)
        else:
            reward = self.compute_scalarized_reward(latency, reliability, energy)
        
        return reward, objectives
    
    def adapt_weights(self, performance: Dict):
        """
        Adapt weights dựa trên performance
        Nếu latency cao → tăng latency weight
        Nếu reliability thấp → tăng reliability weight
        """
        if not self.config.get('multi_objective', {}).get('adaptive_weights', False):
            return
        
        avg_latency = performance.get('avg_latency', 0)
        avg_reliability = performance.get('avg_reliability', 1.0)
        
        # Adjust weights
        if avg_latency > 500:  # High latency
            self.weights.latency = min(0.6, self.weights.latency + 0.05)
            self.weights.reliability = max(0.2, self.weights.reliability - 0.025)
            self.weights.energy = max(0.2, self.weights.energy - 0.025)
        
        if avg_reliability < 0.8:  # Low reliability
            self.weights.reliability = min(0.5, self.weights.reliability + 0.05)
            self.weights.latency = max(0.3, self.weights.latency - 0.025)
            self.weights.energy = max(0.2, self.weights.energy - 0.025)
        
        self.weights.normalize()
    
    def get_stats(self) -> Dict:
        """Lấy statistics về multi-objective optimization"""
        if len(self.pareto_front) == 0:
            return {
                'pareto_solutions': 0,
                'weights': {
                    'latency': self.weights.latency,
                    'reliability': self.weights.reliability,
                    'energy': self.weights.energy
                }
            }
        
        pareto_latencies = [s.latency for s in self.pareto_front]
        pareto_reliabilities = [s.reliability for s in self.pareto_front]
        pareto_energies = [s.energy for s in self.pareto_front]
        
        return {
            'pareto_solutions': len(self.pareto_front),
            'latency_range': [min(pareto_latencies), max(pareto_latencies)],
            'reliability_range': [min(pareto_reliabilities), max(pareto_reliabilities)],
            'energy_range': [min(pareto_energies), max(pareto_energies)],
            'weights': {
                'latency': self.weights.latency,
                'reliability': self.weights.reliability,
                'energy': self.weights.energy
            }
        }

