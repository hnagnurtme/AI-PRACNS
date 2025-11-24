import matplotlib.pyplot as plt
import numpy as np

class NetworkVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def plot_network(self, nodes, connections):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Vẽ các node
        for node_id, node_data in nodes.items():
            pos = node_data.get('position', {})
            x, y = pos.get('longitude', 0), pos.get('latitude', 0)
            node_type = node_data.get('nodeType', 'UNKNOWN')
            
            color = 'green' if node_type == 'GROUND_STATION' else 'blue'
            self.ax.scatter(x, y, c=color, s=100, label=node_type)
            self.ax.annotate(node_id, (x, y))
        
        # Vẽ các kết nối
        for conn in connections:
            node1 = nodes.get(conn['from'])
            node2 = nodes.get(conn['to'])
            if node1 and node2:
                x1, y1 = node1['position']['longitude'], node1['position']['latitude']
                x2, y2 = node2['position']['longitude'], node2['position']['latitude']
                self.ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.5)
        
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_title('SAGIN Network Topology')
        plt.show()