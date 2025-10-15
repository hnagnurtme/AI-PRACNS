import socket
import json
import struct
import sys
import time
import base64
from typing import Dict, Any, Optional
from dataclasses import asdict

# Import data models from the package
from models.app_models import Packet, ServiceQoS
    
# --- HÀM HỖ TRỢ XỬ LÝ DATACLASS CHO JSON ---
def dataclass_to_dict(obj: Any) -> Any:
    """Chuyển đổi dataclass hoặc list dataclass thành dict để JSON.dumps có thể xử lý."""
    if isinstance(obj, list):
        return [dataclass_to_dict(i) for i in obj]
    # Dùng isinstance thay vì hasattr để kiểm tra dataclass an toàn hơn (Python >= 3.7)
    if isinstance(obj, tuple) and hasattr(obj, '__dataclass_fields__'): 
        return asdict(obj)
    if hasattr(obj, '__dataclass_fields__'):
        return asdict(obj)
    return obj

# --- Cấu hình Gửi TCP ---
TIMEOUT_SECONDS = 5.0 

def send_packet_via_tcp(
    target_host: str, 
    target_port: int, 
    packet_object: Packet # Chấp nhận đối tượng Packet (dataclass)
) -> Optional[str]:
    """
    Kết nối đến host:port và gửi đối tượng Packet đã được serialization thành JSON.
    Sử dụng protocol header 4 byte Big-Endian cho độ dài.
    """
    
    try:
        # 1. Serialization: Chuyển đối tượng Packet thành chuỗi JSON
        json_string = json.dumps(packet_object, default=dataclass_to_dict)
        
        json_bytes = json_string.encode('utf-8')
        message_length = len(json_bytes)
        
        # 2. Đóng gói Kích thước (Header: 4 bytes Big-Endian)
        length_header = struct.pack('>I', message_length)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(TIMEOUT_SECONDS)
            s.connect((target_host, target_port))
            
            # 3. Gửi Header và Payload
            s.sendall(length_header)
            s.sendall(json_bytes)
            
            return None 

    except socket.timeout:
        return f"Lỗi Timeout: Không nhận được phản hồi từ {target_host}:{target_port} trong {TIMEOUT_SECONDS}s."
    except ConnectionRefusedError:
        return f"Lỗi Kết nối: Peer {target_host}:{target_port} từ chối kết nối (Listener chưa chạy)."
    except Exception as e:
        return f"Lỗi Socket chung: {e}"

if __name__ == '__main__':
    # --- KHỐI CHẠY ĐỘC LẬP (SỬ DỤNG THAM SỐ DÒNG LỆNH) ---
    
    # 1. Kiểm tra tham số đầu vào
    if len(sys.argv) != 3:
        print("Sử dụng: python TCP_Sender.py <IP_đích> <Port_đích>")
        print("Ví dụ: python TCP_Sender.py 127.0.0.1 50001")
        sys.exit(1)

    try:
        TARGET_IP = sys.argv[1]
        TARGET_PORT = int(sys.argv[2])
        if not (1024 <= TARGET_PORT <= 65535):
             raise ValueError("Cổng phải nằm trong khoảng 1024-65535.")
    except ValueError as e:
        print(f"Lỗi tham số: {e}")
        sys.exit(1)
        
    # If models are not present for standalone run, fall back to minimal definitions.
    try:
        # Try using imported models
        current_time = int(time.time())
        from datetime import datetime
        mock_payload = f"Message sent at {datetime.fromtimestamp(current_time)}"
        mock_packet = Packet(
            packetId=f"TEST-{current_time}",
            sourceUserId="CLI_Sender",
            destinationUserId="Remote_Peer",
            type="DATA",
            serviceQoS=ServiceQoS(maxLatencyMs=100.0, defaultPriority=2),
            payloadDataBase64=base64.b64encode(mock_payload.encode()).decode(),
            payloadSizeByte=len(mock_payload.encode()),
            isUseRL=True
        )
    except Exception:
        # Minimal fallback dataclasses for direct CLI usage
        from dataclasses import dataclass, field

        @dataclass
        class _ServiceQoS:
            maxLatencyMs: float = 150.0
            defaultPriority: int = 1

        @dataclass
        class _Packet:
            packetId: str
            sourceUserId: str
            destinationUserId: str
            type: str = "DATA"
            serviceQoS: _ServiceQoS = field(default_factory=_ServiceQoS)
            payloadDataBase64: str = ""
            payloadSizeByte: int = 0
            TTL: int = 10
            isUseRL: bool = False

        from datetime import datetime
        current_time = int(time.time())
        mock_payload = f"Message sent at {datetime.fromtimestamp(current_time)}"
        mock_packet = _Packet(
            packetId=f"TEST-{current_time}",
            sourceUserId="CLI_Sender",
            destinationUserId="Remote_Peer",
            type="DATA",
            serviceQoS=_ServiceQoS(maxLatencyMs=100.0, defaultPriority=2),
            payloadDataBase64=base64.b64encode(mock_payload.encode()).decode(),
            payloadSizeByte=len(mock_payload.encode()),
            isUseRL=True
        )
    
    print("-" * 50)
    print(f"CHUẨN BỊ GỬI GÓI TIN:")
    print(f"  Đích: {TARGET_IP}:{TARGET_PORT}")
    print(f"  ID: {mock_packet.packetId}")
    print(f"  Payload: '{mock_payload[:30]}...'")
    print("-" * 50)
    
    # Gửi gói tin
    error = send_packet_via_tcp(TARGET_IP, TARGET_PORT, mock_packet)
    
    if error is None:
        print(f"\n✅ Gửi thành công! Gói tin đã được gửi đến {TARGET_IP}:{TARGET_PORT}.")
    else:
        print(f"\n❌ Gửi thất bại: {error}")