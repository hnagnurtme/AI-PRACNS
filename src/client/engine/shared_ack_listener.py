# client_app/engine/shared_ack_listener.py
import socketserver
import threading
import json
from collections import deque
from typing import Dict, Any , Tuple, Optional
import socket

# --- DỮ LIỆU ĐƯỢC CHIA SẺ TRONG BỘ NHỚ ---
ACK_RESULTS_DICT: Dict[str, Dict[str, Any]] = {} 
RECEIVED_DATA_QUEUE: deque = deque(maxlen=100) 

class MultiPurposeTCPRequestHandler(socketserver.BaseRequestHandler):
    """Xử lý mọi gói tin đến (ACK hoặc DATA)."""
    
    def handle(self):
        # Thiết lập timeout cho request socket
        self.request.settimeout(2)
        
        try:
            # Nhận dữ liệu
            self.data = self.request.recv(4096).strip().decode('utf-8')
            if not self.data: return
            
            packet_dict = json.loads(self.data)
            packet_type = packet_dict.get('type')

            # --- Phân loại và Lưu trữ ---
            if packet_type == 'ACK':
                ack_id = packet_dict.get('acknowledgedPacketId')
                if ack_id:
                    # Lưu kết quả ACK để Runner truy cập
                    ACK_RESULTS_DICT[ack_id] = packet_dict
                
            elif packet_type == 'DATA':
                # Lưu gói DATA để UI hiển thị
                RECEIVED_DATA_QUEUE.append(packet_dict)
            
        # BẮT LỖI ĐÃ SỬA: Bắt lỗi socket.timeout trực tiếp
        except socket.timeout:
            # Lỗi xảy ra nếu client kết nối nhưng không gửi dữ liệu trong 2 giây
            print("[Listener Warning] Socket read timeout.")
            pass
        except json.JSONDecodeError:
            # Lỗi nếu dữ liệu nhận được không phải JSON hợp lệ
            print("[Listener Error] Received malformed JSON.")
            pass
        except Exception as e:
            # Bắt các lỗi chung khác
            print(f"[Listener Error] Unexpected error: {e}")
            pass

class SharedACKDataListener(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Server TCP đa luồng lắng nghe chung."""
    allow_reuse_address = True
    
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.ack_results = ACK_RESULTS_DICT
        self.received_data_queue = RECEIVED_DATA_QUEUE

def start_listening_thread(port) -> Tuple[Optional['SharedACKDataListener'], Optional[threading.Thread]]:
    """Khởi động Server Listener trong một Thread nền, trả về Server và Thread."""
    try:
        host, port = '', port
        server = SharedACKDataListener((host, port), MultiPurposeTCPRequestHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        
        # Trả về Tuple (Server, Thread)
        return server, server_thread 
        
    except Exception:
        # Khi có lỗi, trả về Tuple (None, None)
        return None, None