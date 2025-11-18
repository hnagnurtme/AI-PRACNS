
import json
import math
import heapq
import sys
import os
from typing import List, Dict, Any, Tuple, Optional

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python.utils.db_connector import MongoConnector


R_EARTH = 6371.0  # Earth radius in kilometers

def geo_to_ecef(pos: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Converts geographic coordinates (latitude, longitude, altitude) to ECEF (x, y, z).
    Latitude and longitude are in degrees, altitude in km.
    """
    lat = math.radians(pos['latitude'])
    lon = math.radians(pos['longitude'])
    alt = pos.get('altitude', 0)

    N = R_EARTH

    x = (N + alt) * math.cos(lat) * math.cos(lon)
    y = (N + alt) * math.cos(lat) * math.sin(lon)
    z = (N + alt) * math.sin(lat)

    return x, y, z

def distance_3d(pos1_ecef: Tuple[float,float,float], pos2_ecef: Tuple[float,float,float]) -> float:
    """
    Calculates the 3D Euclidean distance between two points in ECEF coordinates.
    """
    x1, y1, z1 = pos1_ecef
    x2, y2, z2 = pos2_ecef
    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)


class DijkstraService:
    def __init__(self, db_connector: Optional[MongoConnector] = None):
        """
        Initialize DijkstraService with database connector.
        If no connector provided, creates a new one.
        """
        self.db = db_connector if db_connector else MongoConnector()

    def build_graph_from_db(self) -> Dict[str, Dict[str, float]]:
        """
        Builds a graph from database nodes.
        The graph is an adjacency list where the weight of the edges is the distance between nodes.
        """
        nodes = self.db.get_all_nodes()

        graph = {node["nodeId"]: {} for node in nodes}
        node_info_cache = {node["nodeId"]: node for node in nodes}

        for node_id, node_data in node_info_cache.items():
            if not node_data.get("isOperational", True):
                continue

            node1_ecef = geo_to_ecef(node_data["position"])

            # If neighbors are specified, use them
            if "neighbors" in node_data and node_data["neighbors"]:
                for neighbor_id in node_data["neighbors"]:
                    if neighbor_id in node_info_cache:
                        neighbor_data = node_info_cache[neighbor_id]
                        if neighbor_data.get("isOperational", True):
                            neighbor_ecef = geo_to_ecef(neighbor_data["position"])
                            distance = distance_3d(node1_ecef, neighbor_ecef)
                            graph[node_id][neighbor_id] = distance
            else:
                # Fallback to calculating neighbors if not present
                node_range = node_data.get("communication", {}).get("maxRangeKm", 2000.0)
                for other_node_id, other_node_data in node_info_cache.items():
                    if node_id == other_node_id or not other_node_data.get("isOperational", True):
                        continue

                    other_node_ecef = geo_to_ecef(other_node_data["position"])
                    distance = distance_3d(node1_ecef, other_node_ecef)

                    if distance <= node_range:
                        graph[node_id][other_node_id] = distance

        return graph

    def build_graph_from_json(self, json_path: str) -> Dict[str, Dict[str, float]]:
        """
        Builds a graph from a JSON file containing a list of nodes.
        The graph is an adjacency list where the weight of the edges is the distance between nodes.
        """
        with open(json_path, 'r') as f:
            nodes = json.load(f)

        graph = {node["nodeId"]: {} for node in nodes}
        node_info_cache = {node["nodeId"]: node for node in nodes}

        for node_id, node_data in node_info_cache.items():
            if not node_data.get("isOperational", True):
                continue

            node1_ecef = geo_to_ecef(node_data["position"])

            # If neighbors are specified, use them
            if "neighbors" in node_data and node_data["neighbors"]:
                for neighbor_id in node_data["neighbors"]:
                    if neighbor_id in node_info_cache:
                        neighbor_data = node_info_cache[neighbor_id]
                        if neighbor_data.get("isOperational", True):
                            neighbor_ecef = geo_to_ecef(neighbor_data["position"])
                            distance = distance_3d(node1_ecef, neighbor_ecef)
                            graph[node_id][neighbor_id] = distance
            else:
                # Fallback to calculating neighbors if not present
                node_range = node_data.get("communication", {}).get("maxRangeKm", 2000.0)
                for other_node_id, other_node_data in node_info_cache.items():
                    if node_id == other_node_id or not other_node_data.get("isOperational", True):
                        continue

                    other_node_ecef = geo_to_ecef(other_node_data["position"])
                    distance = distance_3d(node1_ecef, other_node_ecef)

                    if distance <= node_range:
                        graph[node_id][other_node_id] = distance

        return graph

    def find_shortest_path(self, start_node: str, end_node: str) -> Optional[List[str]]:
        """
        Finds the shortest path from start to end node using Dijkstra's algorithm.
        Returns just the path as a list of node IDs.
        """
        graph = self.build_graph_from_db()
        result = self.calculate_shortest_path(graph, start_node, end_node)
        if result:
            return result[0]
        return None

    def calculate_shortest_path(self, graph: Dict[str, Dict[str, float]], start_node: str, end_node: str) -> Optional[Tuple[List[str], float]]:
        """
        Calculates the shortest path between two nodes using Dijkstra's algorithm.
        Returns the path as a list of node IDs and the total distance.
        """
        if start_node not in graph or end_node not in graph:
            return None

        distances = {node: float('inf') for node in graph}
        distances[start_node] = 0
        previous_nodes = {node: None for node in graph}

        priority_queue = [(0, start_node)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            if current_node == end_node:
                break

            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

        path = []
        current = end_node
        while current is not None:
            path.insert(0, current)
            current = previous_nodes[current]

        if path[0] == start_node:
            return path, distances[end_node]
        else:
            return None

if __name__ == '__main__':
    dijkstra_service = DijkstraService()
    
    # The path to the JSON file is relative to this script's location
    json_path = '../helper/network_nodes.json'
    
    try:
        graph = dijkstra_service.build_graph_from_json(json_path)
        
        # Example: Find the shortest path between two ground stations
        start_node = "GS_HANOI"
        end_node = "GS_TOKYO"

        print(f"Calculating shortest path from {start_node} to {end_node}...")
        
        result = dijkstra_service.calculate_shortest_path(graph, start_node, end_node)

        if result:
            path, distance = result
            print(f"Shortest Path: {' -> '.join(path)}")
            print(f"Total Distance: {distance:.2f} km")
        else:
            print(f"No path found from {start_node} to {end_node}.")

    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found. Make sure the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")
