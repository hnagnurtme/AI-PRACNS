### Run 
cd src/satellite-service && mvn clean compile exec:java -Dexec.mainClass="com.sagin.satellite.SatelliteApp"        

### Test
```
import java.io.*;
import java.net.Socket;
import java.util.Scanner;
import java.util.UUID;

public class SatelliteClientConsole {

    public static void main(String[] args) {

        String serverHost = "localhost";
        int serverPort = 6000;
        Scanner scanner = new Scanner(System.in);

        try (Socket socket = new Socket(serverHost, serverPort);
                OutputStream out = socket.getOutputStream();
                BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {

            System.out.print("Source User: ");
            String src = scanner.nextLine();

            System.out.print("Destination User: ");
            String dst = scanner.nextLine();

            System.out.print("Message: ");
            String msg = scanner.nextLine();

            System.out.print("TTL: ");
            int ttl = Integer.parseInt(scanner.nextLine());

            String packetId = UUID.randomUUID().toString();
            int payloadSize = msg.length();

            // Tạo JSON string thủ công
            String jsonPacket = "{"
                    + "\"packetId\":\"" + packetId + "\","
                    + "\"sourceUserId\":\"" + src + "\","
                    + "\"destinationUserId\":\"" + dst + "\","
                    + "\"message\":\"" + msg + "\","
                    + "\"payloadSize\":" + payloadSize + ","
                    + "\"ttl\":" + ttl // CHÚ Ý: phải là "ttl" lowercase
                    + "}";

            // Gửi JSON packet
            out.write((jsonPacket + "\n").getBytes()); // gửi kèm '\n' để server có thể readLine()
            out.flush();

            System.out.println("Packet sent: " + jsonPacket);

            // Đọc ACK từ server an toàn
            String response = reader.readLine();
            if (response != null) {
                System.out.println("Server response: " + response.trim());
            } else {
                System.out.println("Server response: No response");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        scanner.close();
    }
}
``` 
### Packet Model 
``` 
// ----- Identification -----
    private String packetId;           // ID duy nhất của packet
    private String sourceUserId;       // User gửi
    private String destinationUserId;  // User nhận
    private long timestamp;            // Thời điểm tạo packet (millis)

    // ----- Payload -----
    private String message;            // Nội dung tin nhắn
    private int payloadSize;           // Kích thước dữ liệu

    // ----- Routing / Forwarding -----
    private int TTL;                   // Time-To-Live
    private String currentNode;        // Node đang giữ packet
    private String nextHop;            // Node sẽ gửi tiếp
    private List<String> pathHistory;  // Lịch sử các node đã đi qua

    // ----- QoS / Metrics -----
    private double delayMs;            // Tổng delay từ nguồn
    private double lossRate;           // Tỷ lệ mất packet
    private int retryCount;            // Số lần retry nếu link chưa sẵn sàng
    private int priority;              // Mức ưu tiên (1 cao nhất)
    
    // ----- Status -----
    private boolean dropped;           // Packet đã bị drop chưa
```