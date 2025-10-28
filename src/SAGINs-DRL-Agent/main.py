# main.py (unchanged except for output in process_packets)
import asyncio
import queue
import random
import socket
import json
import threading
import logging
import signal
from typing import Dict, Any
from simulator.network_simulator import Simulator
from env.packet import Packet
from data.mongo_manager import MongoManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_uri = "mongodb://user:password123@localhost:27017/?authSource=admin"

def tcp_server(sim: Simulator, port: int = 8080):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', port))
    server.listen(5)
    logger.info(f"TCP server listening on port {port}")
    while True:
        client, addr = server.accept()
        logger.info(f"Accepted connection from {addr}")
        data = client.recv(1024).decode()
        try:
            request = json.loads(data)
            packet = Packet.from_dict(request['packet'], sim.mongo)
            routed_packet = asyncio.run(sim.route_packet(packet))
            response = json.dumps(routed_packet.to_dict())
            client.send(response.encode())
        except Exception as e:
            logger.error(f"TCP Error parsing JSON: {e}")
            client.send(json.dumps({"error": str(e)}).encode())
        client.close()

async def generate_packets(packet_queue: queue.Queue):
    mongo = MongoManager(mongo_uri)
    service_types = ["VIDEO_STREAM", "AUDIO_CALL", "IMAGE_TRANSFER", "TEXT_MESSAGE"]
    qos_configs = {
        "VIDEO_STREAM": {"defaultPriority": 1, "maxLatencyMs": 150.0, "maxJitterMs": 30.0, "minBandwidthMbps": 5.0, "maxLossRate": 0.01},
        "AUDIO_CALL": {"defaultPriority": 2, "maxLatencyMs": 100.0, "maxJitterMs": 20.0, "minBandwidthMbps": 2.0, "maxLossRate": 0.005},
        "IMAGE_TRANSFER": {"defaultPriority": 1, "maxLatencyMs": 500.0, "maxJitterMs": 50.0, "minBandwidthMbps": 10.0, "maxLossRate": 0.01},
        "TEXT_MESSAGE": {"defaultPriority": 1, "maxLatencyMs": 1000.0, "maxJitterMs": 100.0, "minBandwidthMbps": 0.5, "maxLossRate": 0.05}
    }
    
    # Get all ground stations
    all_gs = [node for node in mongo.get_all_nodes() if node['nodeType'] == 'GROUND_STATION']
    if len(all_gs) < 2:
        logger.error("Need at least 2 ground stations for meaningful routing")
        return
    
    while True:
        service_type = random.choice(service_types)
        qos = qos_configs[service_type]
        
        # Generate TWO DIFFERENT random positions to force cross-GS routing
        client_a_pos = {
            'latitude': random.uniform(-90, 90),
            'longitude': random.uniform(-180, 180),
            'altitude': random.uniform(0, 100)
        }
        
        # Get source GS
        station_source = mongo.get_closest_gs(client_a_pos)
        
        # Force destination to be DIFFERENT GS
        max_attempts = 10
        for attempt in range(max_attempts):
            client_b_pos = {
                'latitude': random.uniform(-90, 90),
                'longitude': random.uniform(-180, 180),
                'altitude': random.uniform(0, 100)
            }
            station_dest = mongo.get_closest_gs(client_b_pos)
            
            if station_dest != station_source:
                break
            
            # If same, move client_b far away
            client_b_pos['latitude'] = -client_a_pos['latitude'] + random.uniform(-20, 20)
            client_b_pos['longitude'] = client_a_pos['longitude'] + 180 + random.uniform(-20, 20)
            if client_b_pos['longitude'] > 180:
                client_b_pos['longitude'] -= 360
            station_dest = mongo.get_closest_gs(client_b_pos)
            
            if station_dest != station_source:
                break
        
        logger.info(f"Generated client_a_pos: {client_a_pos}, client_b_pos: {client_b_pos}")
        logger.info(f"station_source: {station_source}, station_dest: {station_dest}")
        
        if station_source == station_dest:
            logger.warning("⚠️ Still same GS after retry, skipping this packet")
            await asyncio.sleep(3)
            continue
        
        packet = Packet(
            source_user=f"USER_{random.randint(1, 1000)}",
            dest_user=f"USER_{random.randint(1000, 2000)}",
            service_type=service_type,
            payload=f"Sample payload for {service_type}",
            qos=qos,
            client_a_pos=client_a_pos,
            client_b_pos=client_b_pos,
            mongo_manager=mongo
        )
        packet_queue.put(packet)
        await asyncio.sleep(random.uniform(3, 6))

async def process_packets(sim: Simulator, packet_queue: queue.Queue):
    while True:
        if not packet_queue.empty():
            packet = packet_queue.get()
            logger.info(f"Processing packet {packet.packet_id}")
            routed_packet = await sim.route_packet(packet)
            # Enhanced output as per request
            logger.info(f"Packet {routed_packet.packet_id} from {routed_packet.source_user_id} to {routed_packet.destination_user_id}: "
                        f"Path: {routed_packet.path_history}, "
                        f"Bandwidth: {routed_packet.min_bandwidth_mbps} Mbps (min along path), "
                        f"Latency: {routed_packet.accumulated_delay_ms} ms (total), "
                        f"Loss Rate: {routed_packet.accumulated_loss_rate:.4f} (cumulative)")
        await asyncio.sleep(0.1)

def signal_handler(sim: Simulator, sig, frame):
    logger.info("Stopping RL server, saving checkpoint...")
    sim.agent.save_checkpoint()
    sys.exit(0)

async def main():
    sim = Simulator(mongo_uri)
    sim.agent.load_checkpoint()  # Load if exists
    packet_queue = queue.Queue()

    # Signal handler for checkpoint
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sim, sig, frame))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sim, sig, frame))

    # Start TCP server in thread
    tcp_thread = threading.Thread(target=tcp_server, args=(sim,))
    tcp_thread.daemon = True
    tcp_thread.start()

    await asyncio.gather(
        process_packets(sim, packet_queue),
        generate_packets(packet_queue)
    )

if __name__ == "__main__":
    import sys
    asyncio.run(main())