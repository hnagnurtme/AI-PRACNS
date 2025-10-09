package com.sagin.util;

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
}
