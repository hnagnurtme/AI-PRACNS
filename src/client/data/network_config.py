# client_app/data/network_config.py
from models.node import Node, LinkMetric

DEST_SERVER = {"ip": "127.0.0.1", "port": 5000}
CLIENT_LISTEN_PORT = 5001

NETWORK_NODES_DATA = [
    Node(id="G01", type="GROUND", label="Client (Source)", pos=(100, 400), ip="127.0.0.1", port=CLIENT_LISTEN_PORT),
    Node(id="A01", type="AIR", label="HAP-01", pos=(250, 200), processing_ms=1.0),
    Node(id="S01", type="SPACE", label="LEO-01", pos=(400, 50), processing_ms=5.0),
    Node(id="G02", type="GROUND", label="Server (Dest)", pos=(600, 400), ip=DEST_SERVER["ip"], port=DEST_SERVER["port"]),
]

NETWORK_LINKS_DATA = [
    LinkMetric(source="G01", target="A01", distance_km=100, base_latency_ms=10),
    LinkMetric(source="A01", target="S01", distance_km=1000, base_latency_ms=50),
    LinkMetric(source="S01", target="G02", distance_km=800, base_latency_ms=45),
    LinkMetric(source="G01", target="G02", distance_km=500, base_latency_ms=100), 
    LinkMetric(source="A01", target="G02", distance_km=300, base_latency_ms=30),
]

def get_config():
    return {
        "nodes": NETWORK_NODES_DATA,
        "links": NETWORK_LINKS_DATA,
        "dest_server": DEST_SERVER,
        "client_listen_port": CLIENT_LISTEN_PORT
    }