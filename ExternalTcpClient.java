// KHÔNG CÓ PACKAGE - Đây là file Client độc lập
import java.io.PrintWriter;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.Socket;
import java.io.IOException;

/**
 * Ứng dụng Client mô phỏng thiết bị người dùng cuối (UE)
 * gửi dữ liệu TCP thô đến Node Gateway (GS_001) đang chạy trong Docker.
 */
public class ExternalTcpClient {

    // Cổng này PHẢI khớp với cổng bạn đã mở trong docker-compose.yml
    private static final int GATEWAY_PORT = 8080;
    // Hostname là 'localhost' vì bạn đã mở cổng ra máy chủ của mình
    private static final String GATEWAY_HOST = "localhost"; 

    public static void main(String[] args) {
        System.out.println("--- Bắt đầu mô phỏng Client ngoài (TCP) ---");
        
        try (
            // 1. Mở Socket kết nối đến Node Gateway
            Socket socket = new Socket(GATEWAY_HOST, GATEWAY_PORT);
            
            // 2. Thiết lập luồng gửi dữ liệu
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            
            // 3. Thiết lập luồng nhận phản hồi (Không bắt buộc, nhưng tốt cho TCP)
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))
        ) {
            System.out.printf("Đã kết nối thành công đến %s:%d%n", GATEWAY_HOST, GATEWAY_PORT);
            
            // --- GIAI ĐOẠN 1: GỬI DỮ LIỆU ĐỂ KÍCH HOẠT PACKET ---
            
            System.out.println("Gửi 3 gói tin kích hoạt...");
            
            for (int i = 1; i <= 3; i++) {
                String message = "TCP_DATA_CHUNK_" + i;
                out.println(message); // Gửi một dòng dữ liệu (được đọc bởi readLine() trong Gateway)
                System.out.printf("[SENT] Kích hoạt gói tin #%d: %s%n", i, message);
                Thread.sleep(500); // Tạm dừng 500ms
            }

            System.out.println("Hoàn tất gửi. Đóng kết nối...");

        } catch (IOException e) {
            System.err.println("LỖI GIAO TIẾP: Đảm bảo Node GS_001 đang chạy và cổng 8080 đã được mở.");
            System.err.println("Chi tiết: " + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}