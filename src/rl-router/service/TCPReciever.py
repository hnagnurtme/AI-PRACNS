import socket
import json
import sys
import os
import time
import random
import math
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import signal
import psutil
from collections import Counter

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.Packet import Packet, QoS, AnalysisData, HopRecord, Position, BufferState, RoutingDecisionInfo, RoutingAlgorithm
from service.TCPSender import send_packet
from service.DijkstraService import DijkstraService
from service.BatchPacketService import BatchPacketService
from python.utils.db_connector import MongoConnector
from python.utils.state_builder import StateBuilder

# Try to import RL components (optional)
try:
    from python.rl_agent.dqn_model import DuelingDQN, INPUT_SIZE
    RL_AVAILABLE = True
except ImportError:
    print("Warning: RL model not available. Will use fallback routing.")
    RL_AVAILABLE = False
    DuelingDQN = None
    INPUT_SIZE = 0

OUTPUT_SIZE = 10

class QoSMonitor:
    """Monitor and enforce QoS requirements for packets"""
    
    def __init__(self, db_connector: MongoConnector):
        self.db = db_connector
        self.violation_stats = {
            "latency_violations": 0,
            "loss_rate_violations": 0,
            "bandwidth_violations": 0,
            "ttl_expired": 0,
            "node_congestion": 0,
            "node_unavailable": 0
        }
    
    def check_qos_violation(self, packet: Packet, current_node_data: Dict, next_hop_data: Dict = None) -> Optional[str]:
        """
        Check if packet violates QoS requirements and should be dropped
        Returns drop reason if violation occurs, None otherwise
        """
        # Check TTL
        if packet.ttl <= 0:
            self.violation_stats["ttl_expired"] += 1
            return "TTL expired"
        
        # Check if current node is operational
        if not current_node_data.get('isOperational', True):
            self.violation_stats["node_unavailable"] += 1
            return f"Node {current_node_data.get('nodeId')} is not operational"
        
        # Check accumulated latency against max acceptable
        if packet.accumulated_delay_ms > packet.max_acceptable_latency_ms:
            self.violation_stats["latency_violations"] += 1
            return f"Latency violation: {packet.accumulated_delay_ms:.2f}ms > {packet.max_acceptable_latency_ms}ms"
        
        # Check node congestion (buffer overflow)
        current_buffer_usage = current_node_data.get('currentPacketCount', 0)
        buffer_capacity = current_node_data.get('packetBufferCapacity', 1000)
        buffer_utilization = current_buffer_usage / buffer_capacity if buffer_capacity > 0 else 0
        
        if buffer_utilization > 0.95:  # 95% buffer full
            self.violation_stats["node_congestion"] += 1
            return f"Node congestion: {buffer_utilization:.1%} buffer usage"
        
        # If we have next hop data, perform additional checks
        if next_hop_data:
            # Check bandwidth availability for next hop
            current_bandwidth = current_node_data.get('communication', {}).get('bandwidthMHz', 0)
            next_bandwidth = next_hop_data.get('communication', {}).get('bandwidthMHz', 0)
            available_bandwidth = min(current_bandwidth, next_bandwidth)
            
            if available_bandwidth < packet.service_qos.min_bandwidth_mbps:
                self.violation_stats["bandwidth_violations"] += 1
                return f"Bandwidth violation: {available_bandwidth:.2f}MHz < {packet.service_qos.min_bandwidth_mbps}MHz"
            
            # Check if next hop is operational
            if not next_hop_data.get('isOperational', True):
                self.violation_stats["node_unavailable"] += 1
                return f"Next hop node {next_hop_data.get('nodeId')} is not operational"

            # Estimate packet loss probability for the next hop
            # This check is done here because loss occurs on the link to the next hop
            current_hop_loss_rate = current_node_data.get('packetLossRate', 0)
            cumulative_loss_rate = self._calculate_cumulative_loss_rate(packet)
            
            # Correctly calculate total cumulative loss
            total_estimated_loss = 1 - ((1 - cumulative_loss_rate) * (1 - current_hop_loss_rate))
            
            if total_estimated_loss > packet.max_acceptable_loss_rate:
                self.violation_stats["loss_rate_violations"] += 1
                return f"Loss rate violation: {total_estimated_loss:.4f} > {packet.max_acceptable_loss_rate}"
        
        return None  # No violation
    
    def _calculate_cumulative_loss_rate(self, packet: Packet) -> float:
        """Calculate cumulative packet loss rate based on path history"""
        if not packet.hop_records:
            return 0.0
        
        # Use geometric mean for loss rates (more accurate for probabilities)
        loss_rates = [hr.packet_loss_rate for hr in packet.hop_records]
        product = 1.0
        for rate in loss_rates:
            product *= (1 - rate)
        
        return 1 - product
    
    def get_violation_stats(self) -> Dict[str, int]:
        """Get current QoS violation statistics"""
        return self.violation_stats.copy()
    
    def reset_stats(self):
        """Reset violation statistics"""
        self.violation_stats = {
            "latency_violations": 0,
            "loss_rate_violations": 0,
            "bandwidth_violations": 0,
            "ttl_expired": 0,
            "node_congestion": 0,
            "node_unavailable": 0
        }

class PacketLogger:
    """Handle packet logging to database with detailed drop reasons"""
    
    def __init__(self, db_connector: MongoConnector):
        self.db = db_connector
        self.packet_collection = "packet_logs"
    
    def log_packet_drop(self, packet: Packet, drop_reason: str, current_node_id: str):
        """Log packet drop event to database"""
        drop_record = {
            "packet_id": packet.packet_id,
            "source_user_id": packet.source_user_id,
            "destination_user_id": packet.destination_user_id,
            "current_node_id": current_node_id,
            "drop_reason": drop_reason,
            "drop_timestamp": datetime.now(timezone.utc).isoformat(),
            "accumulated_latency_ms": packet.accumulated_delay_ms,
            "hop_count": len(packet.hop_records),
            "path_history": packet.path_history,
            "qos_requirements": {
                "max_latency_ms": packet.max_acceptable_latency_ms,
                "max_loss_rate": packet.max_acceptable_loss_rate,
                "min_bandwidth_mbps": packet.service_qos.min_bandwidth_mbps
            },
            "algorithm_used": "RL" if packet.use_rl else "DIJKSTRA",
            "final_analysis": self._create_final_analysis(packet),
            "status": "DROPPED"
        }
        
        try:
            # Insert into packet_logs collection
            collection = self.db.db[self.packet_collection]
            result = collection.insert_one(drop_record)
            print(f"📝 Packet drop logged to database: {packet.packet_id} - {drop_reason}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Error logging packet drop: {e}")
            return None
    
    def log_packet_delivery(self, packet: Packet, delivery_node_id: str):
        """Log successful packet delivery to database"""
        delivery_record = {
            "packet_id": packet.packet_id,
            "source_user_id": packet.source_user_id,
            "destination_user_id": packet.destination_user_id,
            "delivery_node_id": delivery_node_id,
            "delivery_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_latency_ms": packet.accumulated_delay_ms,
            "total_hops": len(packet.hop_records),
            "final_path": packet.path_history,
            "algorithm_used": "RL" if packet.use_rl else "DIJKSTRA",
            "qos_achieved": {
                "actual_latency_ms": packet.accumulated_delay_ms,
                "estimated_loss_rate": self._calculate_cumulative_loss_rate(packet),
                "success_status": "DELIVERED"
            },
            "final_analysis": self._create_final_analysis(packet),
            "status": "DELIVERED"
        }
        
        try:
            collection = self.db.db[self.packet_collection]
            result = collection.insert_one(delivery_record)
            print(f"✅ Packet delivery logged to database: {packet.packet_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Error logging packet delivery: {e}")
            return None
    
    def _create_final_analysis(self, packet: Packet) -> Dict[str, Any]:
        """Create comprehensive analysis of packet journey"""
        if not packet.hop_records:
            return {}
        
        latencies = [hr.latency_ms for hr in packet.hop_records]
        distances = [hr.distance_km for hr in packet.hop_records]
        loss_rates = [hr.packet_loss_rate for hr in packet.hop_records]
        node_loads = [hr.node_load_percent or 0 for hr in packet.hop_records]
        
        return {
            "total_latency_ms": sum(latencies),
            "total_distance_km": sum(distances),
            "avg_latency_per_hop": sum(latencies) / len(latencies),
            "avg_distance_per_hop": sum(distances) / len(distances),
            "max_latency_hop": max(latencies) if latencies else 0,
            "min_latency_hop": min(latencies) if latencies else 0,
            "avg_loss_rate": sum(loss_rates) / len(loss_rates) if loss_rates else 0,
            "avg_node_load": sum(node_loads) / len(node_loads) if node_loads else 0,
            "hop_by_hop_details": [
                {
                    "hop_number": i + 1,
                    "from_node": hr.from_node_id,
                    "to_node": hr.to_node_id,
                    "latency_ms": hr.latency_ms,
                    "distance_km": hr.distance_km,
                    "loss_rate": hr.packet_loss_rate,
                    "node_load": hr.node_load_percent or 0
                } for i, hr in enumerate(packet.hop_records)
            ]
        }
    
    def _calculate_cumulative_loss_rate(self, packet: Packet) -> float:
        """Calculate cumulative packet loss rate"""
        if not packet.hop_records:
            return 0.0
        
        loss_rates = [hr.packet_loss_rate for hr in packet.hop_records]
        product = 1.0
        for rate in loss_rates:
            product *= (1 - rate)
        
        return 1 - product

def deserialize_packet(data: dict) -> Packet:
    """
    Deserializes a dictionary into a Packet object, handling nested dataclasses.
    """
    # Handle nested dataclasses by manually converting dicts to objects
    if 'service_qos' in data and isinstance(data['service_qos'], dict):
        data['service_qos'] = QoS(**data['service_qos'])

    if 'analysis_data' in data and isinstance(data['analysis_data'], dict):
        data['analysis_data'] = AnalysisData(**data['analysis_data'])

    if 'hop_records' in data and data['hop_records'] is not None:
        hop_records = []
        for hr_data in data['hop_records']:
            if 'from_node_position' in hr_data and hr_data['from_node_position'] is not None:
                hr_data['from_node_position'] = Position(**hr_data['from_node_position'])
            if 'to_node_position' in hr_data and hr_data['to_node_position'] is not None:
                hr_data['to_node_position'] = Position(**hr_data['to_node_position'])
            if 'from_node_buffer_state' in hr_data and hr_data['from_node_buffer_state'] is not None:
                hr_data['from_node_buffer_state'] = BufferState(**hr_data['from_node_buffer_state'])
            if 'routing_decision_info' in hr_data and hr_data['routing_decision_info'] is not None:
                if 'algorithm' in hr_data['routing_decision_info']:
                    hr_data['routing_decision_info']['algorithm'] = RoutingAlgorithm(
                        hr_data['routing_decision_info']['algorithm'])
                hr_data['routing_decision_info'] = RoutingDecisionInfo(**hr_data['routing_decision_info'])
            hop_records.append(HopRecord(**hr_data))
        data['hop_records'] = hop_records

    return Packet(**data)

def calculate_distance_km(pos1: Position, pos2: Position) -> float:
    """
    Calculate 3D Euclidean distance between two positions in km.
    """
    R_EARTH = 6371.0  # km

    lat1_rad = math.radians(pos1.latitude)
    lon1_rad = math.radians(pos1.longitude)
    lat2_rad = math.radians(pos2.latitude)
    lon2_rad = math.radians(pos2.longitude)

    R1 = R_EARTH + pos1.altitude
    R2 = R_EARTH + pos2.altitude

    x1 = R1 * math.cos(lat1_rad) * math.cos(lon1_rad)
    y1 = R1 * math.cos(lat1_rad) * math.sin(lon1_rad)
    z1 = R1 * math.sin(lat1_rad)

    x2 = R2 * math.cos(lat2_rad) * math.cos(lon2_rad)
    y2 = R2 * math.cos(lat2_rad) * math.sin(lon2_rad)
    z2 = R2 * math.sin(lat2_rad)

    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

class TCPReceiver:
    def __init__(self, host: str, port: int, model_path: str = "models/checkpoints/dqn_latest.pth"):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

        # Initialize database connector and services
        self.db = MongoConnector()
        self.dijkstra_service = DijkstraService(self.db)
        self.state_builder = StateBuilder(self.db)

        # Initialize QoS monitoring and packet logging
        self.qos_monitor = QoSMonitor(self.db)
        self.packet_logger = PacketLogger(self.db)

        # Initialize BatchPacket service for automatic packet pair tracking
        self.batch_packet_service = BatchPacketService(self.db)

        # Simulation results tracking
        self.simulation_results = []
        self.current_simulation_id = None

        # Load RL model if available
        self.rl_model = None
        if RL_AVAILABLE:
            try:
                if DuelingDQN is not None:
                    self.rl_model = DuelingDQN(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
                else:
                    raise RuntimeError("DuelingDQN is not available. Ensure RL components are properly imported.")
                
                if os.path.exists(model_path):
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    except TypeError:
                        checkpoint = torch.load(model_path, map_location='cpu')

                    if isinstance(checkpoint, dict):
                        if "model_state_dict" in checkpoint:
                            self.rl_model.load_state_dict(checkpoint["model_state_dict"])
                        elif "model" in checkpoint:
                            self.rl_model.load_state_dict(checkpoint["model"])
                        else:
                            # Try to load directly
                            self.rl_model.load_state_dict(checkpoint)
                    else:
                        self.rl_model.load_state_dict(checkpoint)

                    self.rl_model.eval()
                    print(f"✅ RL Model loaded successfully from {model_path}")
                else:
                    print(f"⚠️ Model file not found at {model_path}. Using fallback routing.")
                    self.rl_model = None
            except Exception as e:
                print(f"⚠️ Error loading RL model: {e}. Using fallback routing.")
                self.rl_model = None

    def listen(self):
        """Start listening for incoming connections"""
        self.sock.listen(5)
        print(f"🚀 TCP Receiver started on {self.host}:{self.port}")
        print(f"📊 RL Model: {'Loaded ✅' if self.rl_model else 'Not Available ⚠️'}")
        print(f"🔍 QoS Monitoring: Active ✅")
        print(f"📝 Packet Logging: Active ✅")
        print(f"💾 Database: Connected ✅")
        print(f"📦 BatchPacket Auto-Save: Active ✅")
        print("-" * 60)
        
        while True:
            try:
                conn, addr = self.sock.accept()
                print(f"🔗 New connection from {addr}")
                client_thread = self._handle_client_async(conn, addr)
            except Exception as e:
                print(f"❌ Error accepting connection: {e}")

    def _handle_client_async(self, conn, addr):
        """Handle client connection in a simple synchronous way (can be enhanced with threading)"""
        try:
            self.handle_client(conn)
        except Exception as e:
            print(f"❌ Error handling client {addr}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()
            print(f"🔌 Connection closed from {addr}")

    def handle_client(self, conn):
        """Handle individual client connection"""
        conn.settimeout(30.0)  # 30 seconds timeout
        data = b""
        
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                
                # Prevent memory exhaustion
                if len(data) > 10 * 1024 * 1024:  # 10MB limit
                    raise ValueError("Packet too large")
        except socket.timeout:
            print("⏰ Client timeout")
            return
        except Exception as e:
            print(f"❌ Client error: {e}")
            return

        if not data:
            print("📭 No data received.")
            return

        try:
            packet_data = json.loads(data.decode('utf-8'))
            packet = deserialize_packet(packet_data)
            self._handle_packet(packet)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to decode JSON: {e}")
        except (TypeError, KeyError) as e:
            print(f"❌ Error deserializing packet: {e}")
            import traceback
            traceback.print_exc()

    def _handle_packet(self, packet: Packet):
        """
        Internal method to process and forward a packet with QoS checking
        """
        print("\n" + "="*80)
        print("📦 PACKET RECEIVED")
        print("="*80)
        print(f"📋 Packet ID: {packet.packet_id}")
        print(f"📍 From: {packet.source_user_id} (Station: {packet.station_source})")
        print(f"🎯 To: {packet.destination_user_id} (Station: {packet.station_dest})")
        print(f"🏠 Current Holder: {packet.current_holding_node_id}")
        print(f"⏱️  Accumulated Latency: {packet.accumulated_delay_ms:.2f}ms")
        print(f"🔄 TTL: {packet.ttl}")
        print(f"🤖 Using RL: {packet.use_rl}")
        print(f"📊 QoS Requirements: Latency<{packet.max_acceptable_latency_ms}ms, "
              f"Loss<{packet.max_acceptable_loss_rate}, "
              f"BW>{packet.service_qos.min_bandwidth_mbps}Mbps")

        # Check if packet already dropped
        if packet.dropped:
            print(f"🔴 Packet {packet.packet_id} already dropped: {packet.drop_reason}")
            return

        # Get current node data for initial QoS check
        current_node_data = self.db.get_node(packet.current_holding_node_id)
        if not current_node_data:
            drop_reason = f"Current node {packet.current_holding_node_id} not found in database"
            self._drop_packet(packet, drop_reason)
            return

        # Initial QoS check at current node
        drop_reason = self.qos_monitor.check_qos_violation(packet, current_node_data)
        if drop_reason:
            self._drop_packet(packet, drop_reason)
            return

        if packet.current_holding_node_id == packet.station_dest:
            self.deliver_to_user(packet)
        else:
            self.process_and_forward_packet(packet)

    def process_and_forward_packet(self, packet: Packet):
        """
        Process packet with QoS monitoring and drop if requirements not met
        """
        # Get current node data
        current_node_data = self.db.get_node(packet.current_holding_node_id)
        if not current_node_data:
            drop_reason = f"Current node {packet.current_holding_node_id} not found in database"
            self._drop_packet(packet, drop_reason)
            return

        # Determine next hop
        if packet.use_rl:
            next_hop_id = self.get_rl_next_hop(packet)
            algo = RoutingAlgorithm.REINFORCEMENT_LEARNING
        else:
            next_hop_id = self.get_dijkstra_next_hop(packet)
            algo = RoutingAlgorithm.DIJKSTRA

        if not next_hop_id:
            drop_reason = "No valid next hop found"
            self._drop_packet(packet, drop_reason)
            return

        # Get next hop node data for QoS checking
        next_hop_data = self.db.get_node(next_hop_id)
        if not next_hop_data:
            drop_reason = f"Next hop node {next_hop_id} not found in database"
            self._drop_packet(packet, drop_reason)
            return

        # 🔴 CRITICAL: Check QoS requirements before forwarding
        drop_reason = self.qos_monitor.check_qos_violation(packet, current_node_data, next_hop_data)
        if drop_reason:
            self._drop_packet(packet, drop_reason)
            return

        print(f"✅ QoS check passed for packet {packet.packet_id}")
        print(f"🔄 Next hop: {next_hop_id} (using {algo.name})")

        # Create Position objects
        from_pos_dict = current_node_data.get('position', {})
        to_pos_dict = next_hop_data.get('position', {})

        from_node_pos = Position(
            latitude=from_pos_dict.get('latitude', 0.0),
            longitude=from_pos_dict.get('longitude', 0.0),
            altitude=from_pos_dict.get('altitude', 0.0)
        )
        to_node_pos = Position(
            latitude=to_pos_dict.get('latitude', 0.0),
            longitude=to_pos_dict.get('longitude', 0.0),
            altitude=to_pos_dict.get('altitude', 0.0)
        )

        # Calculate distance and latency
        distance = calculate_distance_km(from_node_pos, to_node_pos)
        propagation_delay = (distance / 299792.458) * 1000  # ms
        processing_delay = current_node_data.get('nodeProcessingDelayMs', 1.0)
        jitter = random.uniform(0.5, 2.0)
        latency = propagation_delay + processing_delay + jitter

        # Get packet loss rate from current node
        packet_loss_rate = current_node_data.get('packetLossRate', 0.0)

        # Create buffer state
        from_buffer_state = BufferState(
            queue_size=current_node_data.get('currentPacketCount', 0),
            bandwidth_utilization=current_node_data.get('resourceUtilization', 0.0)
        )

        # Create comprehensive hop record
        new_hop_record = HopRecord(
            from_node_id=packet.current_holding_node_id,
            to_node_id=next_hop_id,
            latency_ms=latency,
            timestamp_ms=time.time() * 1000,
            distance_km=distance,
            packet_loss_rate=packet_loss_rate,
            from_node_position=from_node_pos,
            to_node_position=to_node_pos,
            from_node_buffer_state=from_buffer_state,
            routing_decision_info=RoutingDecisionInfo(
                algorithm=algo,
                metric="distance",
                reward=None
            ),
            scenario_type=current_node_data.get('scenarioType', 'NORMAL'),
            node_load_percent=current_node_data.get('resourceUtilization', 0.0) * 100,
            drop_reason_details=None
        )

        # Update packet state
        packet.add_hop_record(new_hop_record)
        packet.accumulated_delay_ms += latency
        packet.current_holding_node_id = next_hop_id
        packet.next_hop_node_id = ""
        packet.ttl -= 1

        print(f"🚀 Forwarding packet to {next_hop_id}")
        print(f"   📏 Distance: {distance:.2f} km")
        print(f"   ⏱️  Latency: {latency:.2f} ms")
        print(f"   📉 Packet Loss Rate: {packet_loss_rate:.4f}")
        print(f"   📊 New Accumulated Latency: {packet.accumulated_delay_ms:.2f}ms")
        print(f"   🏷️  Remaining TTL: {packet.ttl}")

        # Continue processing
        self._handle_packet(packet)

    def _drop_packet(self, packet: Packet, drop_reason: str):
        """
        Handle packet dropping with detailed logging
        """
        packet.dropped = True
        packet.drop_reason = drop_reason

        print(f"\n🔴 PACKET DROPPED")
        print("="*50)
        print(f"📦 Packet ID: {packet.packet_id}")
        print(f"❌ Reason: {drop_reason}")
        print(f"🏠 Final Node: {packet.current_holding_node_id}")
        print(f"⏱️  Accumulated Latency: {packet.accumulated_delay_ms:.2f}ms")
        print(f"🔄 Total Hops: {len(packet.hop_records)}")
        print(f"📊 Final Path: {' → '.join(packet.path_history)}")
        print("="*50)

        # Log to database
        self.packet_logger.log_packet_drop(packet, drop_reason, packet.current_holding_node_id)

        # ✅ AUTO-SAVE to TwoPacket and BatchPacket collections
        self.batch_packet_service.save_packet(packet)

        # Also save to simulation results
        self._save_simulation_result(packet, "DROPPED")

    def deliver_to_user(self, packet: Packet):
        """
        Deliver packet to user with successful logging
        """
        print(f"\n🎯 PACKET REACHED DESTINATION")
        print("="*50)

        # Look up destination user
        dest_user = self.db.get_user(packet.destination_user_id)

        if not dest_user:
            print(f"⚠️ User {packet.destination_user_id} not found in database")
            self.calculate_final_metrics(packet)
            self._save_simulation_result(packet, "USER_NOT_FOUND")
            return

        # Calculate final metrics
        self.calculate_final_metrics(packet)

        # Log successful delivery to database
        log_id = self.packet_logger.log_packet_delivery(packet, packet.current_holding_node_id)

        # ✅ AUTO-SAVE to TwoPacket and BatchPacket collections (SUCCESSFUL DELIVERY)
        self.batch_packet_service.save_packet(packet)

        # Send to user
        user_address = {
            'host': dest_user.get('ipAddress', 'localhost'),
            'port': dest_user.get('port', 10000)
        }

        print(f"👤 Delivering to user: {dest_user.get('userName')}")
        print(f"🌐 Address: {user_address['host']}:{user_address['port']}")

        try:
            send_packet(packet, user_address['host'], user_address['port'])
            print("✅ Packet delivered successfully to user")
            self._save_simulation_result(packet, "DELIVERED")
        except Exception as e:
            print(f"❌ Error delivering packet to user: {e}")
            self._save_simulation_result(packet, "DELIVERY_FAILED")

    def get_rl_next_hop(self, packet: Packet) -> Optional[str]:
        """
        Gets the next hop using integrated RL model
        """
        print("🤖 Using RL for routing (Integrated DQN)")

        if not self.rl_model:
            print("⚠️ RL model not available, using fallback (random neighbor)")
            return self.get_fallback_next_hop(packet)

        try:
            # Prepare packet data for state builder
            packet_state = {
                'currentHoldingNodeId': packet.current_holding_node_id,
                'stationDest': packet.station_dest,
                'path': packet.path_history,
                'ttl': packet.ttl,
                'dropped': packet.dropped,
                'accumulatedDelayMs': packet.accumulated_delay_ms,
                'serviceQoS': {
                    'maxLatencyMs': packet.max_acceptable_latency_ms
                }
            }

            # Get state vector from database
            state_vector = self.state_builder.get_state_vector(packet_state)
            state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)

            # Get neighbors from database
            current_node = self.db.get_node(packet.current_holding_node_id)
            if not current_node:
                return None

            neighbor_ids = current_node.get('neighbors', [])

            # If no neighbors are explicitly defined, find them by proximity
            if not neighbor_ids:
                print("⚠️ No explicit neighbors found. Calculating by proximity...")
                all_nodes = self.db.get_all_nodes()
                current_node_pos_data = current_node.get('position')
                if not current_node_pos_data:
                    return None # Cannot calculate distance without position

                current_node_pos = Position(
                    latitude=current_node_pos_data.get('latitude', 0.0),
                    longitude=current_node_pos_data.get('longitude', 0.0),
                    altitude=current_node_pos_data.get('altitude', 0.0)
                )
                node_range = current_node.get("communication", {}).get("maxRangeKm", 2000.0)

                for other_node in all_nodes:
                    if other_node['nodeId'] == current_node['nodeId'] or not other_node.get("isOperational", True):
                        continue
                    
                    other_node_pos_data = other_node.get('position')
                    if not other_node_pos_data:
                        continue

                    other_node_pos = Position(
                        latitude=other_node_pos_data.get('latitude', 0.0),
                        longitude=other_node_pos_data.get('longitude', 0.0),
                        altitude=other_node_pos_data.get('altitude', 0.0)
                    )

                    distance = calculate_distance_km(current_node_pos, other_node_pos)
                    if distance <= node_range:
                        neighbor_ids.append(other_node['nodeId'])
            
            neighbor_ids = neighbor_ids[:OUTPUT_SIZE]

            if not neighbor_ids:
                print("❌ No neighbors found, even after proximity check.")
                return None

            # Run RL model inference
            with torch.no_grad():
                q_values_tensor = self.rl_model(state_tensor)
            q_values = q_values_tensor.cpu().numpy().flatten()

            # Create valid action mask
            valid_mask = np.zeros(OUTPUT_SIZE, dtype=bool)
            valid_indices = []

            # Prevent backtracking and loops
            last_node_id = packet.path_history[-2] if len(packet.path_history) >= 2 else None

            for i, neighbor_id in enumerate(neighbor_ids):
                if i >= OUTPUT_SIZE:
                    break
                if neighbor_id != last_node_id and neighbor_id not in packet.path_history:
                    valid_mask[i] = True
                    valid_indices.append(i)

            if valid_indices:
                # Select from valid actions only
                valid_q_values = q_values[valid_mask]
                best_valid_index = np.argmax(valid_q_values)
                best_neighbor_index = valid_indices[best_valid_index]
                next_hop = neighbor_ids[best_neighbor_index]
                print(f"🤖 RL selected: {next_hop} (Q-value: {q_values[best_neighbor_index]:.4f})")
                return next_hop
            else:
                # All neighbors are in path, pick best one anyway
                best_index = np.argmax(q_values[:len(neighbor_ids)])
                next_hop = neighbor_ids[best_index]
                print(f"🤖 RL selected (forced): {next_hop} (Q-value: {q_values[best_index]:.4f})")
                return next_hop

        except Exception as e:
            print(f"❌ Error in RL inference: {e}")
            import traceback
            traceback.print_exc()
            return self.get_fallback_next_hop(packet)

    def get_fallback_next_hop(self, packet: Packet) -> Optional[str]:
        """
        Fallback routing: random neighbor from database
        """
        current_node = self.db.get_node(packet.current_holding_node_id)
        if not current_node:
            return None

        neighbors = current_node.get('neighbors', [])
        if not neighbors:
            return None

        # Try to avoid already visited nodes
        possible_next_hops = [n for n in neighbors if n not in packet.path_history]
        if possible_next_hops:
            next_hop = random.choice(possible_next_hops)
            print(f"🎲 Fallback selected: {next_hop}")
            return next_hop

        # If all visited, pick any neighbor
        next_hop = random.choice(neighbors)
        print(f"🎲 Fallback selected (all visited): {next_hop}")
        return next_hop

    def get_dijkstra_next_hop(self, packet: Packet) -> Optional[str]:
        """
        Gets the next hop using Dijkstra's algorithm from database
        """
        print("🗺️ Using Dijkstra for routing")
        path = self.dijkstra_service.find_shortest_path(
            packet.current_holding_node_id,
            packet.station_dest
        )

        if path and len(path) > 1:
            print(f"🗺️ Dijkstra path: {' → '.join(path)}")
            return path[1]
        else:
            print("❌ No path found by Dijkstra")
            return None

    def calculate_final_metrics(self, packet: Packet):
        """
        Calculates final metrics when the packet reaches its destination
        """
        print("\n" + "="*60)
        print("📊 FINAL PACKET ANALYSIS")
        print("="*60)

        if not packet.hop_records:
            print("❌ No hop records available")
            return

        total_latency = sum(hr.latency_ms for hr in packet.hop_records)
        total_distance = sum(hr.distance_km for hr in packet.hop_records)
        num_hops = len(packet.hop_records)

        avg_latency = total_latency / num_hops if num_hops > 0 else 0
        avg_distance = total_distance / num_hops if num_hops > 0 else 0

        packet.analysis_data.total_latency_ms = total_latency
        packet.analysis_data.total_distance_km = total_distance
        packet.analysis_data.avg_latency = avg_latency
        packet.analysis_data.avg_distance_km = avg_distance
        packet.analysis_data.route_success_rate = 1.0 if not packet.dropped else 0.0

        print(f"📦 Packet ID: {packet.packet_id}")
        print(f"📍 Source: {packet.source_user_id} @ {packet.station_source}")
        print(f"🎯 Destination: {packet.destination_user_id} @ {packet.station_dest}")
        print(f"🤖 Algorithm: {'RL (DQN)' if packet.use_rl else 'Dijkstra'}")
        print(f"\n📈 Performance Metrics:")
        print(f"  ⏱️  Total Latency: {total_latency:.2f} ms")
        print(f"  📏 Total Distance: {total_distance:.2f} km")
        print(f"  🔄 Number of Hops: {num_hops}")
        print(f"  📊 Avg Latency/Hop: {avg_latency:.2f} ms")
        print(f"  📐 Avg Distance/Hop: {avg_distance:.2f} km")
        print(f"  ✅ Success Rate: {packet.analysis_data.route_success_rate:.1%}")
        print(f"\n🛣️  Path Taken:")
        print(f"  {' → '.join(packet.path_history)}")

        print(f"\n🔍 Hop Details:")
        for i, hop in enumerate(packet.hop_records, 1):
            print(f"  🔄 Hop {i}: {hop.from_node_id} → {hop.to_node_id}")
            print(f"     📏 Distance: {hop.distance_km:.2f} km")
            print(f"     ⏱️  Latency: {hop.latency_ms:.2f} ms")
            print(f"     📉 Loss Rate: {hop.packet_loss_rate:.4f}")
            if hop.from_node_buffer_state:
                print(f"     📊 Buffer: Queue={hop.from_node_buffer_state.queue_size}, "
                      f"Util={hop.from_node_buffer_state.bandwidth_utilization:.1%}")
            if hop.node_load_percent:
                print(f"     💪 Node Load: {hop.node_load_percent:.1f}%")

        print("="*60)

    def _save_simulation_result(self, packet: Packet, delivery_status: str):
        """
        Saves simulation result
        """
        simulation_result = {
            "simulationId": f"tcp_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{packet.packet_id}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "packetId": packet.packet_id,
            "sourceUser": packet.source_user_id,
            "destinationUser": packet.destination_user_id,
            "algorithm": "RL" if packet.use_rl else "DIJKSTRA",
            "path": packet.path_history,
            "hopRecords": [self._hop_record_to_dict(hr) for hr in packet.hop_records],
            "totalMetrics": {
                "totalLatencyMs": packet.analysis_data.total_latency_ms,
                "totalDistanceKm": packet.analysis_data.total_distance_km,
                "totalHops": len(packet.hop_records),
                "avgLatency": packet.analysis_data.avg_latency,
                "avgDistanceKm": packet.analysis_data.avg_distance_km,
                "routeSuccessRate": packet.analysis_data.route_success_rate
            },
            "deliveryStatus": delivery_status,
            "deliveryTimestamp": datetime.now(timezone.utc).isoformat(),
            "dropped": packet.dropped,
            "dropReason": packet.drop_reason
        }

        self.simulation_results.append(simulation_result)
        print(f"💾 Simulation result saved (Total: {len(self.simulation_results)})")

    def _hop_record_to_dict(self, hop_record: HopRecord) -> Dict:
        """Convert HopRecord to dictionary"""
        return {
            "from_node_id": hop_record.from_node_id,
            "to_node_id": hop_record.to_node_id,
            "latency_ms": hop_record.latency_ms,
            "timestamp_ms": hop_record.timestamp_ms,
            "distance_km": hop_record.distance_km,
            "packet_loss_rate": hop_record.packet_loss_rate,
            "from_node_position": {
                "latitude": hop_record.from_node_position.latitude,
                "longitude": hop_record.from_node_position.longitude,
                "altitude": hop_record.from_node_position.altitude
            } if hop_record.from_node_position else None,
            "to_node_position": {
                "latitude": hop_record.to_node_position.latitude,
                "longitude": hop_record.to_node_position.longitude,
                "altitude": hop_record.to_node_position.altitude
            } if hop_record.to_node_position else None,
            "from_node_buffer_state": {
                "queue_size": hop_record.from_node_buffer_state.queue_size,
                "bandwidth_utilization": hop_record.from_node_buffer_state.bandwidth_utilization
            } if hop_record.from_node_buffer_state else None,
            "routing_decision_info": {
                "algorithm": hop_record.routing_decision_info.algorithm.value,
                "metric": hop_record.routing_decision_info.metric,
                "reward": hop_record.routing_decision_info.reward
            } if hop_record.routing_decision_info else None,
            "scenario_type": hop_record.scenario_type,
            "node_load_percent": hop_record.node_load_percent,
            "drop_reason_details": hop_record.drop_reason_details
        }

    def save_simulation_results(self, filename: str = "tcp_simulation_results.json"):
        """
        Save all simulation results to JSON file
        """
        try:
            with open(filename, "w") as f:
                json.dump(self.simulation_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(self.simulation_results)} TCP simulation results to {filename}")
        except Exception as e:
            print(f"❌ Error saving simulation results: {e}")

    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics from all simulations
        """
        if not self.simulation_results:
            return {"message": "No simulation results available"}

        total_simulations = len(self.simulation_results)
        rl_results = [r for r in self.simulation_results if r["algorithm"] == "RL"]
        dijkstra_results = [r for r in self.simulation_results if r["algorithm"] == "DIJKSTRA"]
        
        successful = [r for r in self.simulation_results if r["deliveryStatus"] == "DELIVERED"]
        dropped = [r for r in self.simulation_results if r["deliveryStatus"] == "DROPPED"]

        summary = {
            "totalSimulations": total_simulations,
            "successfulDeliveries": len(successful),
            "droppedPackets": len(dropped),
            "successRate": len(successful) / total_simulations if total_simulations > 0 else 0,
            "rlSimulations": len(rl_results),
            "dijkstraSimulations": len(dijkstra_results),
            "averageMetrics": {
                "avgTotalLatency": sum(r["totalMetrics"]["totalLatencyMs"] for r in successful) / len(successful) if successful else 0,
                "avgTotalDistance": sum(r["totalMetrics"]["totalDistanceKm"] for r in successful) / len(successful) if successful else 0,
                "avgHops": sum(r["totalMetrics"]["totalHops"] for r in successful) / len(successful) if successful else 0
            }
        }

        return summary

    def get_qos_stats(self) -> Dict[str, Any]:
        """Get comprehensive QoS monitoring statistics"""
        violation_stats = self.qos_monitor.get_violation_stats()
        total_packets = len(self.simulation_results)
        delivered_packets = len([r for r in self.simulation_results if r["deliveryStatus"] == "DELIVERED"])
        dropped_packets = len([r for r in self.simulation_results if r["deliveryStatus"] == "DROPPED"])
        
        # Analyze drop reasons
        drop_reasons = [r.get("dropReason", "Unknown") for r in self.simulation_results if r.get("deliveryStatus") == "DROPPED"]
        drop_reason_counts = Counter(drop_reasons)
        
        return {
            "total_packets_processed": total_packets,
            "successfully_delivered": delivered_packets,
            "dropped_packets": dropped_packets,
            "delivery_success_rate": delivered_packets / total_packets if total_packets > 0 else 0,
            "qos_violation_details": violation_stats,
            "drop_reason_breakdown": dict(drop_reason_counts),
            "most_common_drop_reason": drop_reason_counts.most_common(1)[0][0] if drop_reason_counts else "No drops"
        }

    def health_check(self) -> Dict[str, Any]:
        """Check system health status"""
        return {
            "database_connected": self.db is not None,
            "rl_model_loaded": self.rl_model is not None,
            "dijkstra_service": self.dijkstra_service is not None,
            "qos_monitor_active": self.qos_monitor is not None,
            "active_simulations": len(self.simulation_results),
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def reset_statistics(self):
        """Reset all statistics and simulation results"""
        self.simulation_results.clear()
        self.qos_monitor.reset_stats()
        print("🔄 All statistics have been reset")

def main():
    """Main function to start the TCP Receiver"""
    receiver = TCPReceiver('localhost', 10004)

    def signal_handler(_sig, _frame):
        print("\n\n🛑 Shutting down TCP Receiver...")
        
        # Save simulation results
        receiver.save_simulation_results()
        
        # Print comprehensive statistics
        print("\n" + "="*80)
        print("📊 FINAL STATISTICS SUMMARY")
        print("="*80)
        
        # QoS Statistics
        qos_stats = receiver.get_qos_stats()
        print(f"\n🔍 QoS Monitoring Summary:")
        print(json.dumps(qos_stats, indent=2))
        
        # Simulation Summary
        summary = receiver.get_simulation_summary()
        print(f"\n📈 Simulation Summary:")
        print(json.dumps(summary, indent=2))
        
        # Health Check
        health = receiver.health_check()
        print(f"\n❤️  System Health:")
        print(json.dumps(health, indent=2))
        
        print("\n👋 TCP Receiver shutdown complete")
        exit(0)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        receiver.listen()
    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print(f"❌ Fatal error in TCP Receiver: {e}")
        import traceback
        traceback.print_exc()
        signal_handler(None, None)

if __name__ == '__main__':
    main()