from typing import List, Dict, Optional
from .node import Node
from data.mongodb.connection import MongoDBManager
from simulation.dynamics.weather import WeatherModel
from simulation.dynamics.traffic import TrafficModel

class SAGINNetwork:
    """
    Quản lý toàn bộ mạng SAGIN với tính động.
    """

    def __init__(self, db_manager: MongoDBManager,
                 weather_model: Optional[WeatherModel] = None,
                 traffic_model: Optional[TrafficModel] = None):
        self.db_manager = db_manager
        self.nodes: Dict[str, Node] = {}
        self.simulation_time = 0.0
        self.weather_model = weather_model or WeatherModel()
        self.traffic_model = traffic_model or TrafficModel()
        
    def initialize_network(self, config: Dict):
        """Khởi tạo mạng từ database"""
        nodes_data = self.db_manager.get_all_nodes()
        
        for node_data in nodes_data:
            node = Node.from_dict(node_data)  # Chuyển đổi từ dict sang Node object
            self.nodes[node.nodeId] = node
            
        print(f"Initialized network with {len(self.nodes)} nodes")
    
    def update_network_dynamics(self, delta_time: float):
        """Cập nhật toàn bộ mạng với tính động"""
        self.simulation_time += delta_time

        # Lấy thông tin động từ các model
        weather_impact = self.weather_model.update(delta_time)
        traffic_load = self.traffic_model.update(self.simulation_time)
        
        # Cập nhật từng node
        for node in self.nodes.values():
            if node.isOperational:
                node.update_dynamic_parameters(
                    delta_time=delta_time,
                    weather_impact=weather_impact,
                    network_traffic=traffic_load,
                    time_of_day=(self.simulation_time % 86400) / 86400
                )
    
    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)
    
    def get_operational_nodes(self) -> List[Node]:
        return [node for node in self.nodes.values() if node.isOperational]
    
    def save_network_state(self):
        """Lưu trạng thái mạng vào database"""
        snapshot = {
            'timestamp': self.simulation_time,
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        }
        self.db_manager.save_network_snapshot(snapshot)