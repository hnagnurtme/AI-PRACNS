
import numpy as np
from typing import Dict, List, Any,Optional

class FailureModel:
    """
    Manages node and link failures in the network.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.node_failure_prob = self.config.get('node_failure_prob', 0.001)
        self.link_failure_prob = self.config.get('link_failure_prob', 0.005)
        self.failed_nodes = set()
        self.failed_links = set()
        self.nodes = {} # This should be populated from the environment

    def set_nodes(self, nodes: Dict):
        """
        Sets the nodes in the network for the failure model to use.
        """
        self.nodes = nodes

    def update_failures(self):
        """
        Updates the failure status of nodes and links.
        """
        # Simulate node failures
        for node_id, node in self.nodes.items():
            if node_id not in self.failed_nodes:
                if np.random.rand() < self.node_failure_prob:
                    self.failed_nodes.add(node_id)
                    node.isOperational = False
        
        # Simulate link failures (this is a simplified representation)
        # A more detailed model would manage links explicitly.
        # For now, we can just say some links might be down, but this is not
        # directly used by other parts of the simplified simulation.
        # To implement this properly, we would need a list of all active links.

    def is_node_failed(self, node_id: str) -> bool:
        """Checks if a node has failed."""
        return node_id in self.failed_nodes

    def reset(self):
        """Resets the failure model."""
        for node_id in self.failed_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].isOperational = True
        self.failed_nodes.clear()
        self.failed_links.clear()
