import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os
import time

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from service.TCPReciever import TCPReceiver
from model.Packet import Packet, QoS, AnalysisData, RoutingAlgorithm, RoutingDecisionInfo, HopRecord, Position, BufferState

class TestTCPReceiver(unittest.TestCase):

    def setUp(self):
        # Create an instance of TCPReceiver, but don't bind the socket
        with patch('socket.socket'):
            self.receiver = TCPReceiver('localhost', 65432)

        # Mock the database connector and other services
        self.receiver.db = MagicMock()
        self.receiver.dijkstra_service = MagicMock()
        self.receiver.state_builder = MagicMock()
        self.receiver.rl_model = None  # Test Dijkstra/fallback logic

    def create_base_packet(self):
        """Helper to create a default packet for tests."""
        return Packet(
            packet_id="test_packet",
            source_user_id="user_A",
            destination_user_id="user_B",
            station_source="GS_A",
            station_dest="GS_C",
            current_holding_node_id="GS_A",
            path_history=["GS_A"],
            use_rl=False,
            ttl=10,
            analysis_data=AnalysisData(),
            hop_records=[],
            accumulated_delay_ms=0
        )

    def test_deliver_to_user_success(self):
        """Test the final delivery step to the user."""
        packet = self.create_base_packet()
        packet.current_holding_node_id = "GS_C" # Packet is at destination station
        packet.path_history.append("GS_B")
        packet.path_history.append("GS_C")

        self.receiver.db.get_user.return_value = {
            'userName': 'user_B', 'ipAddress': 'localhost', 'port': 10000
        }

        with patch('service.TCPReciever.send_packet') as mock_send_packet:
            self.receiver._handle_packet(packet)
            
            # It should call deliver_to_user, which calls send_packet
            mock_send_packet.assert_called_once()
            args, _ = mock_send_packet.call_args
            sent_packet, host, port = args
            
            self.assertEqual(host, 'localhost')
            self.assertEqual(port, 10000)
            self.assertEqual(sent_packet.packet_id, packet.packet_id)
            
            # Check if final metrics were calculated
            self.assertEqual(sent_packet.analysis_data.route_success_rate, 1.0)
            self.assertTrue(self.receiver.simulation_results)
            self.assertEqual(self.receiver.simulation_results[0]['deliveryStatus'], "DELIVERED")

    def test_full_internal_forwarding_loop(self):
        """
        Tests the entire internal forwarding loop from source to destination.
        GS_A -> GS_B -> GS_C (dest)
        """
        packet = self.create_base_packet()

        # --- Mock Setup ---
        # 1. Database node data
        node_a_data = {
            '_id': 'GS_A', 'nodeId': 'GS_A', 'neighbors': ['GS_B'],
            'position': {'latitude': 0, 'longitude': 0, 'altitude': 0},
            'communication': {'ipAddress': 'localhost', 'port': 1111},
            'nodeProcessingDelayMs': 1.0, 'currentPacketCount': 10, 'resourceUtilization': 0.1
        }
        node_b_data = {
            '_id': 'GS_B', 'nodeId': 'GS_B', 'neighbors': ['GS_C'],
            'position': {'latitude': 1, 'longitude': 1, 'altitude': 0},
            'communication': {'ipAddress': 'localhost', 'port': 2222},
            'nodeProcessingDelayMs': 1.0, 'currentPacketCount': 20, 'resourceUtilization': 0.2
        }
        node_c_data = {
            '_id': 'GS_C', 'nodeId': 'GS_C', 'neighbors': [],
            'position': {'latitude': 2, 'longitude': 2, 'altitude': 0},
            'communication': {'ipAddress': 'localhost', 'port': 3333},
            'nodeProcessingDelayMs': 1.0, 'currentPacketCount': 30, 'resourceUtilization': 0.3
        }
        
        # 2. Dijkstra service path calculation
        self.receiver.dijkstra_service.find_shortest_path.side_effect = [
            ["GS_A", "GS_B", "GS_C"],  # First call from GS_A
            ["GS_B", "GS_C"],          # Second call from GS_B
        ]

        # 3. DB get_node calls
        def get_node_mock(node_id):
            if node_id == 'GS_A': return node_a_data
            if node_id == 'GS_B': return node_b_data
            if node_id == 'GS_C': return node_c_data
            return None
        self.receiver.db.get_node.side_effect = get_node_mock

        # 4. DB get_user for final delivery
        self.receiver.db.get_user.return_value = {
            'userName': 'user_B', 'ipAddress': 'localhost', 'port': 10000
        }

        # --- Execution ---
        with patch('service.TCPReciever.send_packet') as mock_final_send:
            self.receiver._handle_packet(packet)

        # --- Assertions ---
        # 1. Dijkstra was called twice
        self.assertEqual(self.receiver.dijkstra_service.find_shortest_path.call_count, 2)
        self.receiver.dijkstra_service.find_shortest_path.assert_has_calls([
            call("GS_A", "GS_C"),
            call("GS_B", "GS_C")
        ])

        # 2. Final packet was delivered to the user
        mock_final_send.assert_called_once()
        final_packet, _, _ = mock_final_send.call_args[0]

        # 3. Packet state is correct
        self.assertEqual(final_packet.current_holding_node_id, "GS_C")
        self.assertEqual(final_packet.path_history, ["GS_A", "GS_B", "GS_C"])
        self.assertEqual(final_packet.ttl, 8) # Started at 10, 2 hops
        self.assertEqual(len(final_packet.hop_records), 2)
        self.assertGreater(final_packet.accumulated_delay_ms, 0)

        # 4. Hop records are correct
        self.assertEqual(final_packet.hop_records[0].from_node_id, "GS_A")
        self.assertEqual(final_packet.hop_records[0].to_node_id, "GS_B")
        self.assertEqual(final_packet.hop_records[1].from_node_id, "GS_B")
        self.assertEqual(final_packet.hop_records[1].to_node_id, "GS_C")
        
        # 5. Simulation result was saved correctly
        self.assertEqual(len(self.receiver.simulation_results), 1)
        sim_result = self.receiver.simulation_results[0]
        self.assertEqual(sim_result['deliveryStatus'], "DELIVERED")
        self.assertEqual(sim_result['path'], ["GS_A", "GS_B", "GS_C"])


if __name__ == '__main__':
    unittest.main()