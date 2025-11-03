package com.sagin.util;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.Socket;

public class NetworkUtils {
    public static boolean isServiceAvailable(String host, int port, int timeoutMs) {
        System.out.println("Checking service availability on " + host + ":" + port);
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(host, port), timeoutMs);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Lấy địa chỉ IP thực tế của máy đang chạy.
     *
     * @return IP dạng String, hoặc "127.0.0.1" nếu không lấy được
     */
    public static String getLocalIpAddress() {
        try {
            InetAddress localHost = InetAddress.getLocalHost();
            return localHost.getHostAddress();
        } catch (Exception e) {
            System.err.println("Không thể lấy IP thực tế, dùng default 127.0.0.1");
            return "127.0.0.1";
        }
    }
}
