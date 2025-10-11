# client_app/engine/rl_theory_calculator.py
from typing import Dict, List
from collections import deque

def calculate_theory_path(source: str, dest: str, config: Dict) -> Dict:
    """Tính toán Đường đi Lý thuyết (ít Hop nhất) sử dụng BFS."""
    nodes = {n.id: n for n in config['nodes']}
    adj = {n_id: [] for n_id in nodes}
    
    for link in config['links']:
        if link.is_active: 
            adj[link.source].append(link.target)
            adj[link.target].append(link.source) 

    queue = deque([(source, [source])])
    visited = {source}
    shortest_path = []
    
    while queue:
        current, path = queue.popleft()
        
        if current == dest:
            shortest_path = path
            break
        
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
    
    total_latency = 0.0
    
    if shortest_path:
        for i in range(len(shortest_path) - 1):
            n1, n2 = shortest_path[i], shortest_path[i+1]
            link_data = next((l for l in config['links'] if (l.source==n1 and l.target==n2) or (l.source==n2 and l.target==n1)), None)
            if link_data:
                total_latency += link_data.base_latency_ms
    
    return {
        "path": shortest_path,
        "delay_ms": total_latency if shortest_path else 9999.0
    }