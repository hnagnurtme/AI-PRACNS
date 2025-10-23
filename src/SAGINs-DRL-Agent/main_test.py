import socket
import json
import time

class RLClient:
    def __init__(self, host='localhost', port=5000):
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
    
    def send_packet(self, packet):
        try:
            packet_json = json.dumps(packet)
            self.socket.send(packet_json.encode('utf-8'))
            response = self.socket.recv(4096).decode('utf-8')
            return json.loads(response)
        except Exception as e:
            print(f"Error sending packet: {e}")
            return None
    
    def disconnect(self):
        self.socket.close()
        print("Disconnected from RL Server")
    
    def traverse_path(self, initial_packet):
        current_packet = initial_packet.copy()
        while True:
            response = self.send_packet(current_packet)
            if not response:
                break
            
            if 'error' in response:
                print(f"Error: {response['error']}")
                break
            elif 'status' in response and response['status'] == 'at_dest':
                print(f"Reached destination. Final path: {response['path']}")
                break
            else:
                print(f"Current path: {response['path']}, Next hop: {response['nextHopNodeId']}")
                current_packet['currentHoldingNodeId'] = response['nextHopNodeId']
                time.sleep(1)  # Simulate processing delay
            
            # Optional: Stop after a few hops for testing
            # if len(response.get('path', [])) > 5:
            #     break

if __name__ == "__main__":
    # Initialize client
    client = RLClient()
    
    if not client.connect():
        exit(1)
    
    # Example packet
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
    
    # Traverse the path
    client.traverse_path(initial_packet)
    
    # Disconnect
    client.disconnect()