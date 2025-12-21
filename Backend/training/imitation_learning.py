"""
Imitation Learning Module
Học từ expert demonstrations (Dijkstra, A* algorithms)
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
from environment.constants import (
    STRATIFIED_NEAR_RATIO, STRATIFIED_MEDIUM_RATIO, STRATIFIED_FAR_RATIO,
    STRATIFIED_VERY_FAR_RATIO, STRATIFIED_NEAR_DISTANCE_KM,
    STRATIFIED_MEDIUM_DISTANCE_KM, STRATIFIED_FAR_DISTANCE_KM,
    DEMO_SIMILARITY_THRESHOLD, DEMO_LOG_FREQUENCY, M_TO_KM,
    PATH_QUALITY_LONG_THRESHOLD, PATH_QUALITY_MEDIUM_THRESHOLD,
    PATH_QUALITY_SHORT_THRESHOLD, PATH_QUALITY_LONG_MULTIPLIER,
    PATH_QUALITY_MEDIUM_MULTIPLIER, PATH_QUALITY_SHORT_MULTIPLIER,
    PATH_QUALITY_UTIL_HIGH_THRESHOLD, PATH_QUALITY_UTIL_MEDIUM_THRESHOLD,
    PATH_QUALITY_UTIL_LOW_THRESHOLD
)

logger = logging.getLogger(__name__)


class ExpertDemonstration:
    """Expert demonstration từ traditional algorithms"""
    
    def __init__(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        path: List[Dict],
        algorithm: str
    ):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.path = path
        self.algorithm = algorithm
        self.weight = 1.0
        self.category = None


class ImitationLearning:
    """
    Imitation Learning để học từ expert demonstrations
    Sử dụng Behavior Cloning và DAGGER
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        imitation_config = self.config.get('imitation_learning', {})
        
        self.enabled = imitation_config.get('enabled', True)
        self.use_dagger = imitation_config.get('use_dagger', True)
        self.expert_ratio = imitation_config.get('expert_ratio', 0.3)  # 30% expert actions
        self.bc_loss_weight = imitation_config.get('bc_loss_weight', 0.5)
        
        # Expert demonstrations storage
        self.expert_demos = deque(maxlen=1000)
        self.demo_buffer = []
        
        # DAGGER state
        self.dagger_iteration = 0
        self.mixing_ratio = 1.0  # Start with 100% expert
        
        logger.info(f"Imitation Learning initialized (DAGGER: {self.use_dagger})")
    
    def generate_comprehensive_demos(
        self,
        terminals: List[Dict],
        nodes: List[Dict],
        num_demos: int = 500
    ) -> int:
        """Generate diverse expert demonstrations with stratified sampling"""
        logger.info(f"Generating {num_demos} comprehensive expert demonstrations...")
        
        scenarios = []
        terminal_pairs = []
        
        for i, source in enumerate(terminals):
            for j, dest in enumerate(terminals):
                if i != j:
                    distance = self._calculate_terminal_distance(source, dest)
                    terminal_pairs.append((source, dest, distance))
        
        if len(terminal_pairs) == 0:
            logger.warning("No terminal pairs available for demonstrations")
            return 0
        
        near_pairs = [(s, d) for s, d, dist in terminal_pairs if dist < STRATIFIED_NEAR_DISTANCE_KM]
        medium_pairs = [(s, d) for s, d, dist in terminal_pairs 
                        if STRATIFIED_NEAR_DISTANCE_KM <= dist < STRATIFIED_MEDIUM_DISTANCE_KM]
        far_pairs = [(s, d) for s, d, dist in terminal_pairs 
                     if STRATIFIED_MEDIUM_DISTANCE_KM <= dist < STRATIFIED_FAR_DISTANCE_KM]
        very_far_pairs = [(s, d) for s, d, dist in terminal_pairs if dist >= STRATIFIED_FAR_DISTANCE_KM]
        
        import random
        
        num_near = int(num_demos * STRATIFIED_NEAR_RATIO)
        num_medium = int(num_demos * STRATIFIED_MEDIUM_RATIO)
        num_far = int(num_demos * STRATIFIED_FAR_RATIO)
        num_very_far = int(num_demos * STRATIFIED_VERY_FAR_RATIO)
        
        for _ in range(num_near):
            if near_pairs:
                scenarios.append((*random.choice(near_pairs), 'near'))
        
        for _ in range(num_medium):
            if medium_pairs:
                scenarios.append((*random.choice(medium_pairs), 'medium'))
            elif near_pairs:
                scenarios.append((*random.choice(near_pairs), 'medium'))
        
        for _ in range(num_far):
            if far_pairs:
                scenarios.append((*random.choice(far_pairs), 'far'))
            elif medium_pairs:
                scenarios.append((*random.choice(medium_pairs), 'far'))
        
        for _ in range(num_very_far):
            if very_far_pairs:
                scenarios.append((*random.choice(very_far_pairs), 'very_far'))
            elif far_pairs:
                scenarios.append((*random.choice(far_pairs), 'very_far'))
        
        while len(scenarios) < num_demos:
            source, dest = random.choice(terminal_pairs)[:2]
            scenarios.append((source, dest, 'random'))
        
        successful_demos = 0
        for i, (source, dest, category) in enumerate(scenarios):
            if (i + 1) % DEMO_LOG_FREQUENCY == 0:
                logger.info(f"Generated {i + 1}/{len(scenarios)} demonstrations...")
            
            demo = self.generate_expert_demonstration(
                source, dest, nodes, algorithm='dijkstra'
            )
            
            if demo:
                quality = self._calculate_path_quality(demo.path, nodes)
                demo.weight = quality
                demo.category = category
                self.add_demonstration(demo)
                successful_demos += 1
        
        logger.info(f"Successfully generated {successful_demos}/{num_demos} demonstrations")
        return successful_demos
    
    def _calculate_terminal_distance(self, source: Dict, dest: Dict) -> float:
        """Calculate distance between two terminals in km"""
        from environment.state_builder import RoutingStateBuilder
        state_builder = RoutingStateBuilder(self.config)
        
        source_pos = source.get('position')
        dest_pos = dest.get('position')
        
        if not source_pos or not dest_pos:
            return float('inf')
        
        distance_m = state_builder._calculate_distance(source_pos, dest_pos)
        return distance_m / M_TO_KM
    
    def _calculate_path_quality(self, path: List[Dict], nodes: List[Dict]) -> float:
        """Calculate path quality score (0.0 to 1.0)"""
        if not path or len(path) < 3:
            return 0.0
        
        quality_score = 1.0
        
        if len(path) > PATH_QUALITY_LONG_THRESHOLD:
            quality_score *= PATH_QUALITY_LONG_MULTIPLIER
        elif len(path) > PATH_QUALITY_MEDIUM_THRESHOLD:
            quality_score *= PATH_QUALITY_MEDIUM_MULTIPLIER
        elif len(path) > PATH_QUALITY_SHORT_THRESHOLD:
            quality_score *= PATH_QUALITY_SHORT_MULTIPLIER
        
        node_map = {n['nodeId']: n for n in nodes}
        for path_item in path:
            if path_item.get('type') == 'node':
                node_id = path_item.get('id')
                node = node_map.get(node_id)
                if node:
                    utilization = node.get('resourceUtilization', 0)
                    if utilization > PATH_QUALITY_UTIL_HIGH_THRESHOLD:
                        quality_score *= PATH_QUALITY_LONG_MULTIPLIER
                    elif utilization > PATH_QUALITY_UTIL_MEDIUM_THRESHOLD:
                        quality_score *= PATH_QUALITY_MEDIUM_MULTIPLIER
                    elif utilization > PATH_QUALITY_UTIL_LOW_THRESHOLD:
                        quality_score *= PATH_QUALITY_SHORT_MULTIPLIER
        
        return max(0.0, min(1.0, quality_score))
    
    def generate_expert_demonstration(
        self,
        source_terminal: Dict,
        dest_terminal: Dict,
        nodes: List[Dict],
        algorithm: str = 'dijkstra'
    ) -> Optional[ExpertDemonstration]:
        """
        Generate expert demonstration sử dụng Dijkstra hoặc A*
        """
        try:
            if algorithm == 'dijkstra':
                path_result = self._dijkstra_path(source_terminal, dest_terminal, nodes)
            elif algorithm == 'astar':
                path_result = self._astar_path(source_terminal, dest_terminal, nodes)
            else:
                return None
            
            if not path_result or len(path_result.get('path', [])) < 3:
                return None
            
            # Convert path thành states và actions
            from environment.routing_env import RoutingEnvironment
            from environment.state_builder import RoutingStateBuilder
            
            env = RoutingEnvironment(nodes=nodes, terminals=[source_terminal, dest_terminal])
            state_builder = RoutingStateBuilder(self.config)
            
            states = []
            actions = []
            rewards = []
            
            # Reset environment
            state, info = env.reset(
                options={
                    'source_terminal_id': source_terminal.get('terminalId'),
                    'dest_terminal_id': dest_terminal.get('terminalId')
                }
            )
            states.append(state)
            
            # Follow expert path
            path_nodes = path_result['path']
            current_node_id = None
            
            for i, path_item in enumerate(path_nodes[1:-1]):  # Skip source and dest
                if path_item.get('type') == 'node':
                    node_id = path_item.get('id')
                    
                    # Find action index for this node
                    filtered_nodes = state_builder._smart_node_filtering(
                        nodes, source_terminal, dest_terminal,
                        env.current_node, list(env.visited_nodes)
                    )
                    
                    action = None
                    for idx, node in enumerate(filtered_nodes):
                        if node.get('nodeId') == node_id:
                            action = idx
                            break
                    
                    if action is not None:
                        next_state, reward, terminated, truncated, step_info = env.step(action)
                        states.append(next_state)
                        actions.append(action)
                        rewards.append(reward)
                        
                        if terminated or truncated:
                            break
            
            if len(states) > 1 and len(actions) > 0:
                demo = ExpertDemonstration(
                    states=states[:-1],  # Exclude final state
                    actions=actions,
                    rewards=rewards,
                    path=path_result['path'],
                    algorithm=algorithm
                )
                return demo
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to generate expert demonstration: {e}")
            return None
    
    def _dijkstra_path(
        self,
        source_terminal: Dict,
        dest_terminal: Dict,
        nodes: List[Dict]
    ) -> Optional[Dict]:
        """Dijkstra algorithm để tìm shortest path"""
        try:
            import heapq
            from environment.state_builder import RoutingStateBuilder
            
            state_builder = RoutingStateBuilder(self.config)
            
            # Find nearest ground stations
            source_gs = self._find_nearest_ground_station(source_terminal, nodes)
            dest_gs = self._find_nearest_ground_station(dest_terminal, nodes)
            
            if not source_gs or not dest_gs:
                return None
            
            # Build graph
            all_nodes = [source_gs, dest_gs] + [
                n for n in nodes
                if n.get('isOperational', True) and
                n.get('nodeId') not in [source_gs.get('nodeId'), dest_gs.get('nodeId')]
            ]
            
            graph = {}
            node_map = {n['nodeId']: n for n in all_nodes}
            
            for node in all_nodes:
                graph[node['nodeId']] = []
                node_pos = node.get('position')
                if not node_pos:
                    continue
                
                for other_node in all_nodes:
                    if node['nodeId'] == other_node['nodeId']:
                        continue
                    
                    other_pos = other_node.get('position')
                    if not other_pos:
                        continue
                    
                    distance = state_builder._calculate_distance(node_pos, other_pos)
                    max_range = node.get('communication', {}).get('maxRangeKm', 2000) * 1000
                    
                    if distance <= max_range * 1.5:
                        # Weight = distance + congestion penalty
                        utilization = other_node.get('resourceUtilization', 0)
                        congestion_penalty = (utilization / 100.0) * distance * 0.1
                        weight = distance + congestion_penalty
                        graph[node['nodeId']].append((other_node['nodeId'], weight))
            
            # Dijkstra
            distances = {node_id: float('inf') for node_id in graph}
            previous = {node_id: None for node_id in graph}
            distances[source_gs['nodeId']] = 0
            pq = [(0, source_gs['nodeId'])]
            
            while pq:
                current_dist, current_id = heapq.heappop(pq)
                
                if current_dist > distances[current_id]:
                    continue
                
                if current_id == dest_gs['nodeId']:
                    break
                
                for neighbor_id, edge_weight in graph.get(current_id, []):
                    new_dist = distances[current_id] + edge_weight
                    if new_dist < distances[neighbor_id]:
                        distances[neighbor_id] = new_dist
                        previous[neighbor_id] = current_id
                        heapq.heappush(pq, (new_dist, neighbor_id))
            
            # Reconstruct path
            path_nodes = []
            current = dest_gs['nodeId']
            while current:
                path_nodes.insert(0, node_map[current])
                current = previous.get(current)
            
            if not path_nodes or path_nodes[0]['nodeId'] != source_gs['nodeId']:
                return None
            
            # Build path result
            path_segments = []
            for node in path_nodes:
                path_segments.append({
                    'type': 'node',
                    'id': node.get('nodeId'),
                    'name': node.get('nodeName', node.get('nodeId')),
                    'position': node.get('position')
                })
            
            return {
                'path': [
                    {'type': 'terminal', 'id': source_terminal.get('terminalId')}
                ] + path_segments + [
                    {'type': 'terminal', 'id': dest_terminal.get('terminalId')}
                ],
                'algorithm': 'dijkstra',
                'totalDistance': distances[dest_gs['nodeId']] / 1000
            }
            
        except Exception as e:
            logger.warning(f"Dijkstra failed: {e}")
            return None
    
    def _astar_path(
        self,
        source_terminal: Dict,
        dest_terminal: Dict,
        nodes: List[Dict]
    ) -> Optional[Dict]:
        """A* algorithm với heuristic"""
        # Similar to Dijkstra but with heuristic
        # For now, use Dijkstra as A* (can be enhanced later)
        return self._dijkstra_path(source_terminal, dest_terminal, nodes)
    
    def _find_nearest_ground_station(
        self,
        terminal: Dict,
        nodes: List[Dict]
    ) -> Optional[Dict]:
        """Tìm ground station gần nhất"""
        terminal_pos = terminal.get('position')
        if not terminal_pos:
            return None
        
        ground_stations = [
            n for n in nodes
            if n.get('nodeType') == 'GROUND_STATION' and n.get('isOperational', True)
        ]
        
        if not ground_stations:
            return None
        
        from environment.state_builder import RoutingStateBuilder
        state_builder = RoutingStateBuilder(self.config)
        
        nearest = min(
            ground_stations,
            key=lambda n: state_builder._calculate_distance(
                terminal_pos, n.get('position', {})
            )
        )
        
        return nearest
    
    def compute_behavior_cloning_loss(
        self,
        agent,
        states: List[np.ndarray],
        expert_actions: List[int]
    ) -> torch.Tensor:
        """
        Compute Behavior Cloning loss
        """
        if len(states) == 0 or len(expert_actions) == 0:
            return torch.tensor(0.0)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(agent.device)
        actions_tensor = torch.LongTensor(expert_actions).to(agent.device)
        
        # Get Q-values from agent
        q_values = agent.q_network(states_tensor)
        
        # Cross-entropy loss
        loss = F.cross_entropy(q_values, actions_tensor)
        
        return loss
    
    def add_demonstration(self, demo: ExpertDemonstration):
        """Thêm demonstration vào buffer"""
        self.expert_demos.append(demo)
    
    def sample_expert_action(
        self,
        state: np.ndarray,
        filtered_nodes: List[Dict],
        current_node: Dict,
        dest_terminal: Dict
    ) -> Optional[int]:
        """
        Sample expert action từ demonstrations
        """
        if len(self.expert_demos) == 0:
            return None
        
        # Find closest demonstration state
        best_demo = None
        best_similarity = -1
        
        for demo in self.expert_demos:
            for i, demo_state in enumerate(demo.states):
                # Handle dimension mismatch: pad or truncate to match
                state_dim = len(state)
                demo_dim = len(demo_state)
                
                if state_dim != demo_dim:
                    # If dimensions don't match, skip this demo
                    # This can happen if state dimension changed after demos were generated
                    continue
                
                # Simple cosine similarity
                similarity = np.dot(state, demo_state) / (
                    np.linalg.norm(state) * np.linalg.norm(demo_state) + 1e-8
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_demo = demo
                    best_idx = i
        
        if best_demo and best_similarity > DEMO_SIMILARITY_THRESHOLD:
            if best_idx < len(best_demo.actions):
                return best_demo.actions[best_idx]
        
        return None
    
    def update_dagger(self, success_rate: float):
        """Update DAGGER mixing ratio"""
        if not self.use_dagger:
            return
        
        # Gradually reduce expert ratio as agent improves
        if success_rate > 0.8:
            self.mixing_ratio = max(0.1, self.mixing_ratio - 0.05)
        elif success_rate > 0.6:
            self.mixing_ratio = max(0.2, self.mixing_ratio - 0.02)
        else:
            self.mixing_ratio = min(1.0, self.mixing_ratio + 0.01)
        
        self.dagger_iteration += 1