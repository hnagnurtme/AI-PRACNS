package com.sagin;

import java.io.InputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

public class SimulationUserClient {
    public static void main(String[] args) {
        int port = 7002; // port lắng nghe packet
        System.out.println("Fake user client đang lắng nghe trên port " + port);

        try (ServerSocket serverSocket = new ServerSocket(port)) {
            while (true) {
                Socket socket = serverSocket.accept();
                new Thread(() -> handleConnection(socket)).start();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void handleConnection(Socket socket) {
        try (InputStream is = socket.getInputStream()) {
            byte[] buffer = new byte[4096];
            int read = is.read(buffer);
            if (read > 0) {
                String packetJson = new String(buffer, 0, read, StandardCharsets.UTF_8);
                System.out.println("Đã nhận packet từ Node:");
                System.out.println(packetJson);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                socket.close();
            } catch (Exception ignored) {}
        }
    }
}
