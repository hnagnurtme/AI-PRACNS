package com.sagin.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.NetworkInterface;
import java.net.Socket;
import java.net.SocketException;
import java.util.Enumeration;

public class NetworkUtils {
    private static final Logger logger = LoggerFactory.getLogger(NetworkUtils.class);

    public static boolean isServiceAvailable(String host, int port, int timeoutMs) {
        logger.debug("Checking service availability on {}:{}", host, port);
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(host, port), timeoutMs);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Detects the actual LAN IP address of the machine.
     * This method tries multiple strategies to find the best non-loopback IP address:
     * 1. Iterate through all network interfaces to find a suitable LAN IP
     * 2. Prefer IPv4 over IPv6
     * 3. Skip loopback and link-local addresses
     * 4. Fall back to InetAddress.getLocalHost() if no suitable interface is found
     * 
     * @return IP address as String, or "127.0.0.1" if detection fails
     */
    public static String getLocalIpAddress() {
        logger.info("🔍 Detecting LAN IP address...");
        
        try {
            // Strategy 1: Try to find a non-loopback IP from network interfaces
            String detectedIp = detectLanIpFromInterfaces();
            if (detectedIp != null) {
                logger.info("✅ LAN IP detected from network interfaces: {}", detectedIp);
                return detectedIp;
            }

            // Strategy 2: Fall back to InetAddress.getLocalHost()
            logger.info("⚠️ No suitable network interface found, trying InetAddress.getLocalHost()...");
            InetAddress localHost = InetAddress.getLocalHost();
            String hostAddress = localHost.getHostAddress();
            
            // Check if this is a loopback address
            if (localHost.isLoopbackAddress() || "127.0.0.1".equals(hostAddress) || "::1".equals(hostAddress)) {
                logger.warn("⚠️ InetAddress.getLocalHost() returned loopback address: {}", hostAddress);
                logger.warn("⚠️ Using fallback 127.0.0.1 - multi-machine setup may not work correctly");
                return "127.0.0.1";
            }
            
            logger.info("✅ LAN IP detected from InetAddress.getLocalHost(): {}", hostAddress);
            return hostAddress;
            
        } catch (Exception e) {
            logger.error("❌ Failed to detect LAN IP address: {}", e.getMessage(), e);
            logger.warn("⚠️ Using fallback 127.0.0.1 - multi-machine setup will NOT work");
            return "127.0.0.1";
        }
    }

    /**
     * Detects LAN IP by iterating through network interfaces.
     * Prioritizes IPv4 addresses and skips loopback/link-local addresses.
     * 
     * @return The best LAN IP address found, or null if none found
     */
    private static String detectLanIpFromInterfaces() {
        try {
            Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
            String candidateIp = null;
            
            while (interfaces.hasMoreElements()) {
                NetworkInterface networkInterface = interfaces.nextElement();
                
                // Skip interfaces that are down or loopback
                if (!networkInterface.isUp() || networkInterface.isLoopback()) {
                    continue;
                }
                
                Enumeration<InetAddress> addresses = networkInterface.getInetAddresses();
                while (addresses.hasMoreElements()) {
                    InetAddress address = addresses.nextElement();
                    
                    // Skip loopback addresses
                    if (address.isLoopbackAddress()) {
                        continue;
                    }
                    
                    // Skip link-local addresses (169.254.x.x or fe80::)
                    if (address.isLinkLocalAddress()) {
                        continue;
                    }
                    
                    // Get the IP as a string
                    String ip = address.getHostAddress();
                    
                    // Prefer IPv4 over IPv6
                    if (address.getAddress().length == 4) {
                        // IPv4 address found - this is preferred
                        logger.debug("Found IPv4 LAN IP: {} on interface {}", ip, networkInterface.getName());
                        return ip;
                    } else {
                        // IPv6 address - save as candidate if we don't find IPv4
                        if (candidateIp == null) {
                            // Clean up IPv6 address (remove %interface suffix if present)
                            int percentIndex = ip.indexOf('%');
                            candidateIp = (percentIndex > 0) ? ip.substring(0, percentIndex) : ip;
                            logger.debug("Found IPv6 LAN IP candidate: {} on interface {}", 
                                candidateIp, networkInterface.getName());
                        }
                    }
                }
            }
            
            // Return IPv6 candidate if no IPv4 was found
            if (candidateIp != null) {
                logger.debug("No IPv4 found, using IPv6 candidate: {}", candidateIp);
            }
            return candidateIp;
            
        } catch (SocketException e) {
            logger.error("Error enumerating network interfaces: {}", e.getMessage());
            return null;
        }
    }
}
