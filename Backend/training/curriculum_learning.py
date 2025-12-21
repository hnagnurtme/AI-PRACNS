"""
Curriculum Learning Module
Train từ scenarios đơn giản đến phức tạp để cải thiện learning efficiency
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque

logger = logging.getLogger(__name__)


class CurriculumLevel:
    """Một level trong curriculum"""
    
    def __init__(
        self,
        level: int,
        name: str,
        difficulty: float,
        max_distance_km: float,
        min_nodes: int,
        max_nodes: int,
        allow_obstacles: bool = False,
        require_qos: bool = False
    ):
        self.level = level
        self.name = name
        self.difficulty = difficulty  # 0.0 - 1.0
        self.max_distance_km = max_distance_km
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.allow_obstacles = allow_obstacles
        self.require_qos = require_qos


class CurriculumScheduler:
    """
    Curriculum Learning Scheduler
    Tự động điều chỉnh difficulty dựa trên performance
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        curriculum_config = self.config.get('curriculum', {})
        
        # Curriculum levels
        self.levels = self._create_levels()
        self.current_level = 0
        self.current_difficulty = 0.0
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.success_rate_history = deque(maxlen=50)
        
        # Advancement criteria
        self.min_success_rate = curriculum_config.get('min_success_rate', 0.7)
        self.min_episodes_at_level = curriculum_config.get('min_episodes_at_level', 100)
        self.episodes_at_current_level = 0
        
        # Adaptive difficulty
        self.adaptive = curriculum_config.get('adaptive', True)
        self.difficulty_increment = curriculum_config.get('difficulty_increment', 0.05)
        
        logger.info(f"Curriculum Learning initialized with {len(self.levels)} levels")
    
    def _create_levels(self) -> List[CurriculumLevel]:
        """Tạo các levels từ đơn giản đến phức tạp"""
        levels = [
            # Level 0: Rất đơn giản - gần, ít nodes, không obstacles
            CurriculumLevel(
                level=0,
                name="Beginner",
                difficulty=0.1,
                max_distance_km=1000,  # 1000km
                min_nodes=5,
                max_nodes=53,  # Match actual operational nodes count
                allow_obstacles=False,
                require_qos=False
            ),
            # Level 1: Đơn giản - gần, nhiều nodes hơn
            CurriculumLevel(
                level=1,
                name="Easy",
                difficulty=0.2,
                max_distance_km=2000,
                min_nodes=10,
                max_nodes=40,  # 🆕 CHANGED: 20 → 40
                allow_obstacles=False,
                require_qos=False
            ),
            # Level 2: Trung bình - xa hơn, nhiều nodes
            CurriculumLevel(
                level=2,
                name="Medium",
                difficulty=0.4,
                max_distance_km=5000,
                min_nodes=20,
                max_nodes=60,  # 🆕 CHANGED: 40 → 60
                allow_obstacles=True,
                require_qos=False
            ),
            # Level 3: Khó - rất xa, nhiều nodes, có QoS
            CurriculumLevel(
                level=3,
                name="Hard",
                difficulty=0.6,
                max_distance_km=10000,
                min_nodes=40,
                max_nodes=77,  # 🆕 CHANGED: 60 → 77 (actual DB size)
                allow_obstacles=True,
                require_qos=True
            ),
            # Level 4: Rất khó - toàn cầu, tất cả nodes, QoS strict
            CurriculumLevel(
                level=4,
                name="Expert",
                difficulty=0.8,
                max_distance_km=20000,
                min_nodes=60,
                max_nodes=81,
                allow_obstacles=True,
                require_qos=True
            ),
            # Level 5: Master - không giới hạn
            CurriculumLevel(
                level=5,
                name="Master",
                difficulty=1.0,
                max_distance_km=float('inf'),
                min_nodes=0,
                max_nodes=81,
                allow_obstacles=True,
                require_qos=True
            ),
        ]
        return levels
    
    def update_performance(
        self,
        success: bool,
        reward: float,
        episode_length: int
    ):
        """Update performance metrics"""
        self.performance_history.append({
            'success': success,
            'reward': reward,
            'length': episode_length
        })
        
        if success:
            self.success_rate_history.append(1.0)
        else:
            self.success_rate_history.append(0.0)
        
        self.episodes_at_current_level += 1
    
    def should_advance(self) -> bool:
        """Kiểm tra xem có nên advance lên level tiếp theo không"""
        if self.current_level >= len(self.levels) - 1:
            return False  # Đã ở level cao nhất
        
        if self.episodes_at_current_level < self.min_episodes_at_level:
            return False  # Chưa đủ episodes
        
        # Tính success rate
        if len(self.success_rate_history) < 20:
            return False
        
        recent_success_rate = np.mean(list(self.success_rate_history)[-20:])
        
        if recent_success_rate >= self.min_success_rate:
            logger.info(
                f"Advancing from level {self.current_level} to {self.current_level + 1} "
                f"(success rate: {recent_success_rate:.2f})"
            )
            return True
        
        return False
    
    def advance_level(self):
        """Advance lên level tiếp theo"""
        if self.current_level < len(self.levels) - 1:
            self.current_level += 1
            self.episodes_at_current_level = 0
            self.success_rate_history.clear()
            logger.info(f"Advanced to level {self.current_level}: {self.levels[self.current_level].name}")
    
    def get_current_level(self) -> CurriculumLevel:
        """Lấy current level"""
        return self.levels[self.current_level]
    
    def filter_scenario(
        self,
        source_terminal: Dict,
        dest_terminal: Dict,
        nodes: List[Dict]
    ) -> Tuple[bool, Optional[str]]:
        """
        Kiểm tra xem scenario có phù hợp với current level không
        Returns: (is_valid, reason)
        """
        level = self.get_current_level()
        
        # Tính khoảng cách
        from environment.state_builder import RoutingStateBuilder
        state_builder = RoutingStateBuilder(self.config)
        
        source_pos = source_terminal.get('position')
        dest_pos = dest_terminal.get('position')
        
        if not source_pos or not dest_pos:
            return False, "Missing position data"
        
        distance = state_builder._calculate_distance(source_pos, dest_pos) / 1000  # Convert to km
        
        # Kiểm tra distance
        if distance > level.max_distance_km:
            return False, f"Distance {distance:.1f}km exceeds max {level.max_distance_km}km"
        
        # Kiểm tra số lượng nodes
        operational_nodes = [n for n in nodes if n.get('isOperational', True)]
        num_nodes = len(operational_nodes)
        
        if num_nodes < level.min_nodes:
            return False, f"Only {num_nodes} nodes, need at least {level.min_nodes}"
        
        if num_nodes > level.max_nodes:
            # Filter nodes để phù hợp với level
            return True, f"Filtering {num_nodes} nodes to {level.max_nodes}"
        
        return True, None
    
    def filter_nodes(self, nodes: List[Dict]) -> List[Dict]:
        """Filter nodes để phù hợp với current level"""
        level = self.get_current_level()
        
        operational_nodes = [n for n in nodes if n.get('isOperational', True)]
        
        if len(operational_nodes) <= level.max_nodes:
            return operational_nodes
        
        # Ưu tiên nodes tốt hơn (ground stations, GEO, MEO, LEO)
        node_priority = {
            'GROUND_STATION': 4,
            'GEO_SATELLITE': 3,
            'MEO_SATELLITE': 2,
            'LEO_SATELLITE': 1
        }
        
        # Sort by priority và quality
        def node_score(node):
            node_type = node.get('nodeType', '')
            priority = node_priority.get(node_type, 0)
            quality = node.get('resourceUtilization', 50)  # Lower is better
            return (priority, -quality)
        
        sorted_nodes = sorted(operational_nodes, key=node_score, reverse=True)
        return sorted_nodes[:level.max_nodes]
    
    def get_difficulty(self) -> float:
        """Lấy current difficulty (0.0 - 1.0)"""
        base_difficulty = self.get_current_level().difficulty
        
        if not self.adaptive:
            return base_difficulty
        
        # Adaptive difficulty dựa trên performance
        if len(self.performance_history) < 10:
            return base_difficulty
        
        recent_rewards = [p['reward'] for p in list(self.performance_history)[-10:]]
        avg_reward = np.mean(recent_rewards)
        
        # Nếu performance tốt, tăng difficulty
        if avg_reward > 10000:
            adaptive_difficulty = min(1.0, base_difficulty + self.difficulty_increment)
        else:
            adaptive_difficulty = base_difficulty
        
        return adaptive_difficulty
    
    def get_stats(self) -> Dict:
        """Lấy statistics về curriculum progress"""
        if len(self.success_rate_history) == 0:
            recent_success_rate = 0.0
        else:
            recent_success_rate = np.mean(list(self.success_rate_history)[-20:])
        
        return {
            'current_level': self.current_level,
            'level_name': self.levels[self.current_level].name,
            'difficulty': self.get_difficulty(),
            'episodes_at_level': self.episodes_at_current_level,
            'recent_success_rate': recent_success_rate,
            'total_levels': len(self.levels)
        }

