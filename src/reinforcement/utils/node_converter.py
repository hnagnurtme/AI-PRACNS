"""
Utility to convert database node dictionaries to Node objects.
"""

from simulation.core.node import Node, Position, Orbit, Velocity, Communication
from typing import Dict, Any


def dict_to_node(node_dict: Dict[str, Any]) -> Node:
    """
    Convert a node dictionary from database to a Node object.

    Args:
        node_dict: Dictionary containing node data

    Returns:
        Node object
    """
    # Extract position
    pos_dict = node_dict.get('position', {})
    position = Position(
        latitude=pos_dict.get('lat', 0.0),
        longitude=pos_dict.get('lon', 0.0),
        altitude=pos_dict.get('alt', 0.0)
    )

    # Create default orbit and velocity
    orbit = Orbit()
    velocity = Velocity()

    # Extract communication
    comm_dict = node_dict.get('communication', {})
    communication = Communication(
        frequencyGHz=comm_dict.get('frequencyGHz', 2.4),
        bandwidthMHz=comm_dict.get('bandwidthMHz', 1000.0),
        transmitPowerDbW=comm_dict.get('transmitPowerDbW', 40.0),
        antennaGainDb=comm_dict.get('antennaGainDb', 10.0),
        beamWidthDeg=comm_dict.get('beamWidthDeg', 45.0),
        maxRangeKm=comm_dict.get('maxRangeKm', 5000.0),
        minElevationDeg=comm_dict.get('minElevationDeg', 10.0),
        ipAddress=comm_dict.get('ipAddress', '0.0.0.0'),
        port=comm_dict.get('port', 8080)
    )

    # Create node
    node = Node(
        nodeId=node_dict.get('node_id', ''),
        nodeName=node_dict.get('node_id', ''),
        nodeType=node_dict.get('node_type', 'UNKNOWN'),
        position=position,
        orbit=orbit,
        velocity=velocity,
        communication=communication,
        isOperational=node_dict.get('isOperational', node_dict.get('operational', True)),
        batteryChargePercent=int(node_dict.get('batteryChargePercent', node_dict.get('battery', 100.0))),
        resourceUtilization=node_dict.get('resourceUtilization', node_dict.get('congestion', 0.0)),
        neighbors=node_dict.get('neighbors', [])
    )

    # Set link quality
    node.communication.link_quality = node_dict.get('link_quality', 1.0)

    return node


def nodes_dict_to_objects(nodes_dict: Dict[str, Dict]) -> Dict[str, Node]:
    """
    Convert a dictionary of node dictionaries to Node objects.

    Args:
        nodes_dict: Dictionary mapping node IDs to node dictionaries

    Returns:
        Dictionary mapping node IDs to Node objects
    """
    return {
        node_id: dict_to_node(node_data)
        for node_id, node_data in nodes_dict.items()
    }
