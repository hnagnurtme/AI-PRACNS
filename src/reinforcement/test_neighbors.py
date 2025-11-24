#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from scripts.setup_database import create_sample_network_data

# Create network
snapshot = create_sample_network_data()

# Check neighbors
total_neighbors = 0
for node_id, node in snapshot['nodes'].items():
    neighbors = node.get('neighbors', [])
    total_neighbors += len(neighbors)
    if len(neighbors) > 0:
        print(f"{node_id} ({node['nodeType']}): {len(neighbors)} neighbors")

print(f"\nTotal nodes: {len(snapshot['nodes'])}")
print(f"Total neighbors computed: {total_neighbors}")
print(f"Average neighbors per node: {total_neighbors / len(snapshot['nodes']):.2f}")
