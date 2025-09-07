package com.sagsins.core.utils;

import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.InterfaceAddress;
import java.net.NetworkInterface;
import java.util.Enumeration;

public class NetworkUtils {

    /**
     * Lấy CIDR mạng cục bộ của máy (IPv4), luôn trả về giá trị hợp lệ.
     * @return network IP + prefix length, ví dụ "192.168.3.0/22"
     */
    public static String getLocalCidr() {
        try {
            Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
            while (interfaces.hasMoreElements()) {
                NetworkInterface iface = interfaces.nextElement();

                if (!iface.isUp() || iface.isLoopback()) continue;

                for (InterfaceAddress addr : iface.getInterfaceAddresses()) {
                    InetAddress inet = addr.getAddress();
                    if (inet instanceof Inet4Address) {
                        int prefix = addr.getNetworkPrefixLength();
                        String network = calculateNetworkAddress(inet.getHostAddress(), prefix);
                        return network + "/" + prefix;
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * Tính địa chỉ network từ IP và prefix length.
     */
    private static String calculateNetworkAddress(String ip, int prefixLength) {
        String[] octets = ip.split("\\.");
        int ipInt = (Integer.parseInt(octets[0]) << 24)
                  | (Integer.parseInt(octets[1]) << 16)
                  | (Integer.parseInt(octets[2]) << 8)
                  | Integer.parseInt(octets[3]);

        int mask = ~((1 << (32 - prefixLength)) - 1);
        int networkInt = ipInt & mask;

        return ((networkInt >> 24) & 0xFF) + "."
             + ((networkInt >> 16) & 0xFF) + "."
             + ((networkInt >> 8) & 0xFF) + "."
             + (networkInt & 0xFF);
    }

}
