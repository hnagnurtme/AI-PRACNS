# client_app/engine/tcp_socket_sender.py
import socket
from typing import Tuple
from ..models.packet import Packet
from ..utils.json_serializer import packet_to_json

def send_data_packet(dest_ip: str, dest_port: int, packet: Packet) -> Tuple[bool, str]:
    """Mở socket client, gửi gói DATA (JSON) đến Server Java, và đóng socket."""
    
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(3)
        client_socket.connect((dest_ip, dest_port))
        
        packet_json = packet_to_json(packet)
        client_socket.sendall((packet_json + '\n').encode('utf-8'))
        
        client_socket.close()
        return True, "DATA packet sent successfully."
    
    except socket.error as e:
        return False, f"SOCKET_ERROR: Failed to connect or send data. {e}"
    except Exception as e:
        return False, f"GENERAL_ERROR: {e}"