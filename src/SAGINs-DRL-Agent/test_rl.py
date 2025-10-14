# test_one_route.py
import torch
import collections
import glob
import os
from data.mongo_manager import MongoDataManager
from env.link_metrics_calculator import LinkMetricsCalculator
from env.state_processor import StateProcessor
from env.action_mapper import ActionMapper
from agents.dpn_agent import DqnAgent

def test_one_route(source: str, destination: str):
    """Test 1 route cụ thể"""
    print(f"🎯 Testing: {source} → {destination}")
    
    mongo_manager = MongoDataManager()
    link_calculator = LinkMetricsCalculator()
    
    # Tìm model .pth
    model_files = glob.glob("models/*.pth")
    if not model_files:
        print("❌ No .pth files found")
        return
    
    model_path = max(model_files, key=os.path.getmtime)
    
    # Dijkstra - chỉ tìm đường đi
    snapshot = mongo_manager.get_training_snapshot()
    nodes = snapshot['nodes']
    
    visited = set()
    queue = collections.deque([(source, [source])])
    visited.add(source)
    
    d_path = []
    while queue:
        current, path = queue.popleft()
        
        if current == destination:
            d_path = path
            break
        
        for neighbor in nodes:
            if neighbor == current or neighbor in visited:
                continue
            
            link_metrics = link_calculator.calculate_link_metrics(
                nodes[current], nodes[neighbor]
            )
            
            if link_metrics.get('isLinkActive', False):
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    print(f"🧮 Dijkstra: {' → '.join(d_path) if d_path else 'No path'}")
    if d_path:
        print(f"   Hops: {len(d_path) - 1}")
    
    # RL - từ .pth
    checkpoint = torch.load(model_path, map_location='cpu')
    state_processor = StateProcessor(max_neighbors=10)
    action_mapper = ActionMapper(mongo_manager)
    
    agent = DqnAgent(
        state_size=state_processor.get_state_size(),
        action_size=action_mapper.get_action_size(),
        learning_rate=0.001,
        gamma=0.99
    )
    
    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
    agent.policy_net.eval()
    agent.epsilon = 0.0
    agent.action_mapper = action_mapper
    
    rl_path = [source]
    current = source
    visited_rl = set([source])
    
    for hop in range(8):
        if current == destination:
            break
        
        current_data = nodes[current]
        available = []
        state_data = {
            "sourceNodeId": current, "destinationNodeId": destination,
            "targetQoS": {"serviceType": "VIDEO_STREAMING"},
            "sourceNodeInfo": current_data,
            "destinationNodeInfo": nodes[destination],
            "neighborLinkMetrics": {}
        }
        
        for neighbor in nodes:
            if neighbor != current and neighbor not in visited_rl:
                link_metrics = link_calculator.calculate_link_metrics(current_data, nodes[neighbor])
                if link_metrics['isLinkActive']:
                    state_data['neighborLinkMetrics'][neighbor] = link_metrics
                    available.append(neighbor)
        
        if not available:
            break
        
        state_vector = state_processor.json_to_state_vector(state_data)
        next_hop = agent.select_action(state_vector)
        
        if next_hop not in available:
            next_hop = available[0]
        
        rl_path.append(next_hop)
        visited_rl.add(next_hop)
        current = next_hop
    
    print(f"🧠 RL: {' → '.join(rl_path) if rl_path else 'No path'}")
    if rl_path:
        success = (rl_path[-1] == destination)
        print(f"   Hops: {len(rl_path) - 1}")
        print(f"   Success: {'✅' if success else '❌'}")
    
    # Kết quả
    print(f"\n📊 KẾT QUẢ:")
    if d_path and rl_path and rl_path[-1] == destination:
        if len(rl_path) < len(d_path):
            print("🎉 RL WIN - Ít hop hơn")
        elif len(rl_path) > len(d_path):
            print("📈 Dijkstra WIN - Ít hop hơn") 
        else:
            print("🤝 TIE - Cùng số hop")
    elif d_path:
        print("📈 Dijkstra WIN - RL failed")
    elif rl_path and rl_path[-1] == destination:
        print("🎉 RL WIN - Dijkstra failed")
    else:
        print("❌ Both failed")

if __name__ == "__main__":
    # Thay đổi source và destination tại đây
    test_one_route("GS_SINGAPORE", "GS_TOKYO")