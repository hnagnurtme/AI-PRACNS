import socket
import json
import struct
import sys
from typing import Optional, Any
import base64
import time

sys.path.insert(0, '.')
try:
    from models.packet import Packet
except ImportError:
    print("Cảnh báo: Không thể import model Packet. Chế độ test sẽ sử dụng dictionary.")
    Packet = None

def send_packet_via_tcp(host: str, port: int, packet: Any) -> Optional[str]:
    """
    Gửi một đối tượng packet đến một địa chỉ TCP sử dụng giao thức header 4-byte.

    Giao thức: [Độ dài Payload (4 bytes, Big-Endian)] + [Payload (JSON bytes)]

    Args:
        host: Địa chỉ IP của máy chủ nhận.
        port: Cổng của máy chủ nhận.
        packet: Đối tượng Packet (hoặc dictionary) để gửi.

    Returns:
        None nếu gửi thành công.
        Một chuỗi chứa thông báo lỗi nếu thất bại.
    """
    try:
        # 1. Chuyển đổi đối tượng Packet thành chuỗi JSON
        # Giả định đối tượng có phương thức to_json() trả về một chuỗi JSON
        if hasattr(packet, 'to_json') and callable(packet.to_json):
            json_string = packet.to_json()
        elif isinstance(packet, dict):
            json_string = json.dumps(packet)
        else:
            return "Lỗi: Dữ liệu đầu vào không phải là Packet object hoặc dictionary."

        # 2. Mã hóa chuỗi JSON thành bytes
        # Ensure json_string is a string before encoding
        if not isinstance(json_string, str):
            json_string = str(json_string)
        json_bytes = json_string.encode('utf-8')

        # 3. Đóng gói header: Tính toán độ dài và tạo header 4-byte
        #    '>' = Big-Endian (Network Order)
        #    'I' = Unsigned Integer (4 bytes)
        header = struct.pack('>I', len(json_bytes))

        # 4. Tạo tin nhắn hoàn chỉnh để gửi
        message = header + json_bytes

        # 5. Mở kết nối, gửi dữ liệu và đóng kết nối
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            sock.sendall(message)
        
        return None # Gửi thành công

    except ConnectionRefusedError:
        return f"Kết nối bị từ chối. Máy chủ tại {host}:{port} không hoạt động hoặc sai địa chỉ."
    except socket.error as e:
        return f"Lỗi socket: {e}"
    except Exception as e:
        return f"Một lỗi không xác định đã xảy ra: {e}"

# --- KHỐI CHẠY ĐỘC LẬP ĐỂ KIỂM TRA ---
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Sử dụng: python service/TCP_Sender.py <host> <port>")
        print("Ví dụ: python service/TCP_Sender.py 127.0.0.1 50001")
        sys.exit(1)

    TARGET_HOST = sys.argv[1]
    try:
        TARGET_PORT = int(sys.argv[2])
    except ValueError:
        print("Lỗi: Port phải là một con số.")
        sys.exit(1)

    # Tạo một gói tin mẫu để gửi
    # Nếu import Packet thành công, dùng object. Nếu không, dùng dict.
    if Packet:
        from models.packet import ServiceQoS
        sample_packet = Packet(
            packetId="PKT-TEST-001",
            sourceUserId="SENDER_TEST",
            destinationUserId="LISTENER_TEST",
            stationSource="GS-TEST-S",
            stationDest="GS-TEST-D",
            type="DATA",
            acknowledgedPacketId=None,
            timeSentFromSourceMs=int(time.time() * 1000),
            payloadDataBase64=base64.b64encode(b"Hello from TCP_Sender test!").decode('utf-8'),
            payloadSizeByte=25,
            serviceType="TEXT_MESSAGE",
            serviceQoS=ServiceQoS(serviceType="TEXT_MESSAGE", defaultPriority=4, maxLatencyMs=2000, maxJitterMs=500, minBandwidthMbps=0.1, maxLossRate=0.05),
            TTL=10,
            currentHoldingNodeId="SENDER_TEST",
            nextHopNodeId="",
            pathHistory=["SENDER_TEST"],
            hopRecords=[],
            priorityLevel=4,
            isUseRL=False
        )
    else:
        # Dictionary dự phòng nếu không import được model
        sample_packet = {
            "packetId": "PKT-DICT-TEST-002",
            "sourceUserId": "SENDER_TEST_DICT",
            "type": "DATA",
            "payloadDataBase64": base64.b64encode(b"Hello from dictionary test!").decode('utf-8')
        }

    print(f"Đang chuẩn bị gửi gói tin đến {TARGET_HOST}:{TARGET_PORT}...")
    
    # Gọi hàm để gửi gói tin
    error = send_packet_via_tcp(TARGET_HOST, TARGET_PORT, sample_packet)
    
    if error:
        print(f"\n❌ Gửi thất bại!")
        print(f"   Lý do: {error}")
    else:
        print("\n✅ Gói tin đã được gửi thành công!")