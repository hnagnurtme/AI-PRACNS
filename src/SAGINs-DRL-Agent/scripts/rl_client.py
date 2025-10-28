import socket
import json

HOST = '127.0.0.1'
PORT = 5050

sample_packet = {
    'stationSource': 'N-HANOI',
    'stationDest': 'N-TOKYO',
    'currentHoldingNodeId': 'N-HANOI',
    'TTL': 10,
    'serviceType': 'VIDEO_STREAM',
    'payloadSizeByte': 512,
    'priorityLevel': 1,
    'accumulatedDelayMs': 0,
}

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.send(json.dumps(sample_packet).encode('utf-8'))
    data = s.recv(65536)
    try:
        resp = json.loads(data.decode('utf-8'))
    except Exception:
        resp = data.decode('utf-8')
    print('Response from server:')
    print(json.dumps(resp, indent=2))
