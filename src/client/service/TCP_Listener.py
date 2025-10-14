import socket
import threading
import time
from typing import Callable, Optional, Dict, Any
import json
import sys
import struct
import os
from models.app_models import Packet  # Import Model from package
try:
    # Import the shared queue so the listener always pushes into the same instance
    from service.incoming_queue import GLOBAL_INCOMING_QUEUE
except Exception:
    GLOBAL_INCOMING_QUEUE = None

# --- Class TCPListener ---

class TCPListener(threading.Thread):
    """
    Service lắng nghe các kết nối TCP đến trên một cổng cụ thể.
    Sử dụng protocol header 4 byte Big-Endian cho độ dài gói tin.
    """

    def __init__(self, host: str, port: int, handler: Callable[[Dict], None]):
        """Khởi tạo Listener."""
        super().__init__()
        self.host = host
        self.port = port
        self.handler = handler
        self.running = True
        self.listener_socket: Optional[socket.socket] = None
        self.name = f"TCP_Listener:{port}"
        self.daemon = True 

    def stop(self):
        """Dừng luồng lắng nghe."""
        self.running = False
        print(f"\n[{self.name}] Đang yêu cầu dừng...")
        
        # Ngắt socket để giải phóng blocking accept()
        if self.listener_socket:
            try:
                # Gửi kết nối giả để giải phóng blocking accept()
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('127.0.0.1', self.port))
                self.listener_socket.close()
            except:
                pass

    def run(self):
        """Hàm chính của luồng: Thiết lập socket và lắng nghe."""
        try:
            self.listener_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            self.listener_socket.bind((self.host, self.port))
            self.listener_socket.listen(5)
            
            print(f"[{self.name}] Đã khởi động và lắng nghe trên {self.host}:{self.port}")

            while self.running:
                try:
                    self.listener_socket.settimeout(0.5) 
                    conn, addr = self.listener_socket.accept()
                    print(f"[{self.name}] Kết nối đến từ: {addr}")
                    
                    threading.Thread(target=self._handle_connection, args=(conn, addr), daemon=True).start()
                    
                except socket.timeout:
                    continue
                except socket.error as e:
                    if self.running:
                        print(f"[{self.name}] Lỗi socket Accept: {e}")
                    break 
                
        except Exception as e:
            if self.running:
                print(f"[{self.name}] LỖI KHỞI TẠO: Không thể bind hoặc listen. {e}")
        finally:
            if self.listener_socket:
                self.listener_socket.close()
            if self.running: 
                # Không in nếu dừng do socket.error, chỉ in nếu dừng bình thường
                pass 
            print(f"[{self.name}] Đã dừng.")

    def _recvall(self, sock: socket.socket, n: int) -> Optional[bytearray]:
        """Hàm trợ giúp đảm bảo nhận đủ n byte."""
        data = bytearray()
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except:
                return None
        return data

    def _handle_connection(self, conn: socket.socket, addr: tuple):
        """Xử lý giao tiếp với client: Nhận dữ liệu theo protocol Header (4 bytes length)."""
        try:
            # 1. Nhận Header (4 bytes)
            header = self._recvall(conn, 4)
            if not header: return

            # 2. Giải nén độ dài payload
            # Dùng '>' cho Big-Endian, 'I' cho Unsigned Integer (4 bytes)
            message_length = struct.unpack('>I', header)[0]
            
            # 3. Nhận Payload (JSON)
            json_bytes = self._recvall(conn, message_length)
            if not json_bytes: return
            
            json_data = json_bytes.decode('utf-8')
            packet_dict = json.loads(json_data)
            
            # Construct the message
            msg = {"source_addr": addr, "packet_data": packet_dict}

            # Prefer a provided handler for backwards compatibility. If a handler
            # is supplied we assume it is responsible for enqueuing or further
            # processing. If no handler is provided, fall back to pushing into
            # the shared GLOBAL_INCOMING_QUEUE (if available).
            try:
                if callable(self.handler):
                    # Call handler and do NOT also push into the queue to avoid duplicates
                    self.handler(msg)
                else:
                    if GLOBAL_INCOMING_QUEUE is not None:
                        GLOBAL_INCOMING_QUEUE.put_nowait(msg)
            except Exception as e:
                # If handler raised, attempt to at least push into the queue as a best-effort
                try:
                    if GLOBAL_INCOMING_QUEUE is not None:
                        GLOBAL_INCOMING_QUEUE.put_nowait(msg)
                except Exception:
                    pass
                print(f"[{self.name}] Error in handler/queue push: {e}")
            
        except json.JSONDecodeError:
            print(f"[{self.name}] Lỗi giải mã JSON từ {addr}: Dữ liệu không phải JSON hợp lệ.")
        except Exception as e:
            print(f"[{self.name}] Lỗi xử lý dữ liệu từ {addr}: {e}")
        finally:
            conn.close()


if __name__ == '__main__':
    # --- KHỐI CHẠY ĐỘC LẬP (SỬ DỤNG THAM SỐ DÒNG LỆNH) ---
    
    # Hàm xử lý đơn giản khi nhận được gói tin
    def mock_packet_handler(received_data: Dict):
        addr = received_data['source_addr']
        packet = received_data['packet_data']
        
        # Cố gắng hiển thị thông tin gói tin
        payload_preview = packet.get('payloadDataBase64', '')
        
        print("-" * 30)
        print(f"NHẬN GÓI TỪ: {addr}")
        print(f"ID: {packet.get('packetId', 'N/A')}")
        print(f"Loại: {packet.get('type', 'N/A')}")
        print(f"Kích thước Payload: {len(payload_preview)} bytes (Base64)")
        print(f"Payload Preview: {payload_preview[:20]}...")
        print("-" * 30)

    # 1. Kiểm tra tham số đầu vào
    if len(sys.argv) != 2:
        print("Sử dụng: python TCP_Listener.py <port>")
        print("Ví dụ: python TCP_Listener.py 50001")
        sys.exit(1)

    try:
        LISTEN_PORT = int(sys.argv[1])
        if not (1024 <= LISTEN_PORT <= 65535):
             raise ValueError("Cổng phải nằm trong khoảng 1024-65535.")
    except ValueError as e:
        print(f"Lỗi tham số cổng: {e}")
        sys.exit(1)

    HOST = '0.0.0.0' # Lắng nghe trên mọi interface
    
    # 2. Khởi động Listener
    listener = TCPListener(HOST, LISTEN_PORT, mock_packet_handler)
    listener.start()
    
    # 3. Giữ luồng chính hoạt động
    try:
        print(f"\n[INFO] Listener đã sẵn sàng. Gửi dữ liệu TCP đến IP máy bạn:{LISTEN_PORT}")
        print("Nhấn Ctrl+C để dừng.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Yêu cầu dừng chương trình...")
    finally:
        listener.stop()
        listener.join()
        print("[INFO] Chương trình đã kết thúc.")