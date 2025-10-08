package com.sagin.util;

import java.net.InetSocketAddress;
import java.net.Socket;

public class NetworkUtils {
    public static boolean isServiceAvailable(String host, int port, int timeoutMs) {
    try (Socket socket = new Socket()) {
        // Cố gắng kết nối đến host:port trong thời gian giới hạn
        socket.connect(new InetSocketAddress(host, port), timeoutMs);
        return true;
    } catch (Exception e) {
        // Không thể kết nối (Connection refused, timeout, host not found)
        return false;
    }
}
}
