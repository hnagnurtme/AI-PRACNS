# p2p_gui_chat.py
import socket
import threading
import sys
from tkinter import *
from tkinter import simpledialog, messagebox, scrolledtext
import ipaddress

# --- Cấu hình Mạng Cố định/Mặc định ---
DEFAULT_PORT = 50001
MY_IP = '0.0.0.0' # Lắng nghe trên mọi interface

class P2PChatApp:
    def __init__(self, master, listen_port):
        self.master = master
        master.title("P2P Chat Client")
        
        self.listen_port = listen_port
        self.running = True
        
        # --- Cài đặt GUI ---
        
        # Frame chính
        main_frame = Frame(master, padx=10, pady=10)
        main_frame.pack(fill=BOTH, expand=True)

        # 1. Hiển thị thông tin Peer của mình
        my_ip = socket.gethostbyname(socket.gethostname())
        self.info_label = Label(main_frame, text=f"ID CỦA BẠN: {my_ip}:{self.listen_port}", 
                                fg="blue", font=("Arial", 10, "bold"))
        self.info_label.pack(fill=X, pady=(0, 10))

        # 2. Hộp thoại hiển thị tin nhắn (Read-only)
        self.chat_history = scrolledtext.ScrolledText(main_frame, state='disabled', height=15, width=60)
        self.chat_history.pack(fill=BOTH, expand=True, pady=5)
        self.log_message(f"--- Đang lắng nghe trên cổng {self.listen_port} ---", "system")

        # 3. Khu vực Gửi tin nhắn (Controls Frame)
        control_frame = Frame(main_frame)
        control_frame.pack(fill=X, pady=(10, 0))
        
        # Host/Port đích
        Label(control_frame, text="IP Đích:").grid(row=0, column=0, sticky=W)
        self.target_ip_entry = Entry(control_frame, width=15)
        self.target_ip_entry.insert(0, '127.0.0.1')
        self.target_ip_entry.grid(row=0, column=1, padx=5, sticky=W)
        
        Label(control_frame, text="Port Đích:").grid(row=0, column=2, sticky=W)
        self.target_port_entry = Entry(control_frame, width=8)
        self.target_port_entry.insert(0, '50001')
        self.target_port_entry.grid(row=0, column=3, padx=5, sticky=W)
        
        # Tin nhắn
        Label(control_frame, text="Tin nhắn:").grid(row=1, column=0, sticky=W)
        self.message_entry = Entry(control_frame, width=40)
        self.message_entry.grid(row=1, column=1, columnspan=3, padx=5, sticky=W)
        self.message_entry.bind('<Return>', lambda event: self.send_message_gui()) # Gửi bằng Enter
        
        # Nút Gửi
        self.send_button = Button(control_frame, text="Gửi", command=self.send_message_gui)
        self.send_button.grid(row=1, column=4, padx=5)

        # Cài đặt sự kiện đóng cửa sổ
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Bắt đầu Luồng Lắng nghe
        self.listener_thread = threading.Thread(target=self.listener_thread_func, daemon=True)
        self.listener_thread.start()

    def log_message(self, message, source="incoming"):
        """Đưa tin nhắn vào hộp thoại lịch sử chat."""
        self.chat_history.config(state='normal')
        
        if source == "incoming":
            tag = "blue"
            prefix = "[NHẬN] "
        elif source == "system":
            tag = "green"
            prefix = "[HỆ THỐNG] "
        else: # outgoing
            tag = "red"
            prefix = "[GỬI] "

        self.chat_history.tag_config(tag, foreground=tag)
        self.chat_history.insert(END, prefix, tag)
        self.chat_history.insert(END, message + '\n')
        self.chat_history.see(END) # Cuộn xuống cuối
        self.chat_history.config(state='disabled')

    def listener_thread_func(self):
        """Luồng Server: Lắng nghe kết nối đến."""
        try:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
            listener.bind((MY_IP, self.listen_port))
            listener.listen(5)
            
            while self.running:
                # Dùng timeout ngắn để kiểm tra self.running thường xuyên
                listener.settimeout(1) 
                try:
                    conn, addr = listener.accept()
                    # Bắt đầu luồng riêng biệt để xử lý tin nhắn
                    threading.Thread(target=self.handle_incoming_message, args=(conn, addr), daemon=True).start()
                except socket.timeout:
                    continue # Quay lại kiểm tra self.running
                
        except Exception as e:
            if self.running:
                self.log_message(f"LỖI LẮNG NGHE: Port {self.listen_port} có thể đã được sử dụng. {e}", "system")
        finally:
            if 'listener' in locals():
                listener.close()
            self.log_message("Luồng lắng nghe đã dừng.", "system")


    def handle_incoming_message(self, conn, addr):
        """Xử lý kết nối và nhận tin nhắn từ một peer cụ thể."""
        try:
            message = conn.recv(1024).decode('utf-8')
            if message:
                self.log_message(f"Từ {addr[0]}:{addr[1]}: {message}", "incoming")
                
        except Exception as e:
            self.log_message(f"Lỗi nhận dữ liệu từ {addr[0]}: {e}", "system")
        finally:
            conn.close()

    def send_message_gui(self):
        """Hành động gửi tin nhắn khi nhấn nút (chạy trong luồng chính)."""
        target_ip = self.target_ip_entry.get().strip()
        target_port_str = self.target_port_entry.get().strip()
        message = self.message_entry.get().strip()
        
        if not all([target_ip, target_port_str, message]):
            messagebox.showerror("Lỗi", "Vui lòng điền đầy đủ IP, Port và Tin nhắn.")
            return

        try:
            target_port = int(target_port_str)
            ipaddress.ip_address(target_ip) # Xác thực IP cơ bản
            
            # Gửi tin nhắn
            self.send_message(target_ip, target_port, message)
            
        except ValueError:
            messagebox.showerror("Lỗi", "Port hoặc IP không hợp lệ.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi không xác định: {e}")
            
    def send_message(self, target_ip, target_port, message):
        """Logic kết nối và gửi dữ liệu."""
        try:
            sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sender.connect((target_ip, target_port))
            sender.sendall(message.encode('utf-8'))
            
            self.log_message(f"Đến {target_ip}:{target_port}: {message}", "outgoing")
            self.message_entry.delete(0, END) # Xóa nội dung sau khi gửi
            
        except ConnectionRefusedError:
            self.log_message(f"Gửi thất bại. Peer {target_ip}:{target_port} không lắng nghe.", "system")
        except Exception as e:
            self.log_message(f"Lỗi gửi tin: {e}", "system")
        finally:
            if 'sender' in locals():
                sender.close()

    def on_closing(self):
        """Xử lý khi đóng cửa sổ."""
        self.running = False
        self.master.destroy()
        sys.exit() # Đảm bảo tất cả các luồng daemon đều kết thúc

def get_listen_port():
    """Hỏi người dùng Port lắng nghe khi khởi chạy."""
    root = Tk()
    root.withdraw() # Ẩn cửa sổ root chính

    # Mặc định sử dụng simpledialog trong Tkinter để lấy input
    while True:
        try:
            port = simpledialog.askinteger("Cấu hình P2P", 
                                          "Nhập Port bạn muốn lắng nghe (ví dụ: 50001):",
                                          initialvalue=DEFAULT_PORT,
                                          minvalue=1024, maxvalue=65535)
            if port is None:
                sys.exit() # Người dùng nhấn Cancel
            return port
        except TypeError:
            messagebox.showerror("Lỗi", "Port phải là một số nguyên hợp lệ.")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

if __name__ == "__main__":
    listen_port = get_listen_port()
    
    root = Tk()
    app = P2PChatApp(root, listen_port)
    root.mainloop()