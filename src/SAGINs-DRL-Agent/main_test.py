import socket
import json
import time

class RLClient:
    # Sửa cổng kết nối mặc định thành 8080
    def __init__(self, host='localhost', port=8080): 
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    def connect(self):
        try:
            self.socket.connect((self.host, self.port))
            print(f"Connected to RL Server at {self.host}:{self.port}")
        except ConnectionRefusedError:
            print("Connection refused. Ensure RL Server is running.")
            return False
        return True
    
    def send_packet(self, packet: dict):
        # Hàm này vẫn giữ nguyên, nhưng nó sẽ được gọi 1 lần
        try:
            packet_json = json.dumps(packet)
            # Thêm newline hoặc ký tự kết thúc nếu server cần (mặc dù server hiện tại không cần)
            self.socket.sendall(packet_json.encode('utf-8')) 
            
            # Đợi phản hồi
            response = self.socket.recv(4096).decode('utf-8')
            return json.loads(response)
        except Exception as e:
            print(f"Error sending packet: {e}")
            return None
    
    def disconnect(self):
        self.socket.close()
        print("Disconnected from RL Server")
    
    # Thay thế hoàn toàn traverse_path bằng hàm send_single_route
    def send_single_route(self, initial_packet: dict):
        print(f"Requesting full route from {initial_packet.get('currentHoldingNodeId')} to {initial_packet.get('destination_node_id', 'N/A')}...")
        
        # Sửa cấu trúc gói tin để khớp với Packet.from_dict
        # Server mong đợi cấu trúc {"packet": { ... nội dung gói tin ... }}
        full_request = {"packet": initial_packet}
        
        response = self.send_packet(full_request)
        
        if not response:
            print("No response received.")
            return
        
        if 'error' in response:
            print(f"Server Error: {response['error']}")
            return

        # Phân tích phản hồi cuối cùng từ Simulator/route_packet
        print("\n--- Route Result ---")
        
        # KIỂM TRA: nodeType có phải là 'LEO_SATELLITE' không
        is_dropped = response.get('dropped', False)
        
        if is_dropped:
            print(f"⚠️ PACKET DROPPED! Reason: {response.get('drop_reason', 'Unknown')}")
        else:
            print(f"✅ Route Success!")
            print(f"  Path History: {response.get('path_history')}")
            print(f"  Total Latency: {response.get('accumulated_delay_ms'):.2f} ms")
            print(f"  Min Bandwidth: {response.get('min_bandwidth_mbps'):.2f} Mbps")
            print(f"  Total Hops: {len(response.get('path_history', [])) - 1}")


if __name__ == "__main__":
    # Initialize client (Mặc định cổng 8080)
    client = RLClient()
    
    if not client.connect():
        exit(1)
    
    # Sửa cấu trúc gói tin để khớp với yêu cầu của lớp Packet
    # destination_node_id phải được tính toán bởi Packet.from_dict()
    # dựa trên client_b_pos, nhưng chúng ta cần cung cấp các trường chính
    initial_packet = {

        "stationSource": "GS-06",

        "stationDest": "GS-01",

        "currentHoldingNodeId": "GS-06",

        "TTL": 10,

        "serviceType": "VIDEO_STREAMING",

        "payloadSizeByte": 512,

        "priorityLevel": 1,

        "accumulatedDelayMs": 0,

        "maxAcceptableLatencyMs": 150,

        "maxAcceptableLossRate": 0.01

    }
    
    # Gửi yêu cầu định tuyến trọn gói
    client.send_single_route(initial_packet)
    
    # Disconnect
    client.disconnect()