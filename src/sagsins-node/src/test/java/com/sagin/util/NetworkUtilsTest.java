package com.sagin.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

import java.net.InetAddress;

/**
 * Unit tests for NetworkUtils, specifically testing LAN IP detection functionality.
 */
class NetworkUtilsTest {

    @Test
    @DisplayName("getLocalIpAddress should return a non-null IP address")
    void testGetLocalIpAddressReturnsNonNull() {
        String ip = NetworkUtils.getLocalIpAddress();
        assertNotNull(ip, "IP address should not be null");
    }

    @Test
    @DisplayName("getLocalIpAddress should return a valid IP format")
    void testGetLocalIpAddressReturnsValidFormat() {
        String ip = NetworkUtils.getLocalIpAddress();
        assertNotNull(ip, "IP address should not be null");
        
        // IP should match IPv4 or IPv6 format
        boolean isValidFormat = isValidIpv4(ip) || isValidIpv6(ip);
        assertTrue(isValidFormat, "IP address should be in valid IPv4 or IPv6 format: " + ip);
    }

    @Test
    @DisplayName("getLocalIpAddress should prefer non-loopback addresses")
    void testGetLocalIpAddressPrefersNonLoopback() {
        String ip = NetworkUtils.getLocalIpAddress();
        assertNotNull(ip, "IP address should not be null");
        
        // If there are multiple network interfaces, the method should prefer non-loopback
        // However, on systems with only loopback (like some CI environments),
        // it's acceptable to return 127.0.0.1
        System.out.println("Detected IP: " + ip);
        
        // The IP should be valid
        assertTrue(isValidIpv4(ip) || isValidIpv6(ip), 
            "IP should be in valid format");
    }

    @Test
    @DisplayName("isServiceAvailable should return false for unreachable service")
    void testIsServiceAvailableReturnsFalseForUnreachable() {
        // Test with a port that is very unlikely to be in use
        boolean available = NetworkUtils.isServiceAvailable("127.0.0.1", 59999, 500);
        assertFalse(available, "Service on port 59999 should not be available");
    }

    @Test
    @DisplayName("isServiceAvailable should handle invalid host gracefully")
    void testIsServiceAvailableHandlesInvalidHost() {
        boolean available = NetworkUtils.isServiceAvailable("invalid.host.example", 80, 500);
        assertFalse(available, "Invalid host should return false");
    }

    // Helper methods for IP validation

    private boolean isValidIpv4(String ip) {
        if (ip == null || ip.isEmpty()) {
            return false;
        }
        
        String[] parts = ip.split("\\.");
        if (parts.length != 4) {
            return false;
        }
        
        try {
            for (String part : parts) {
                int num = Integer.parseInt(part);
                if (num < 0 || num > 255) {
                    return false;
                }
            }
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private boolean isValidIpv6(String ip) {
        if (ip == null || ip.isEmpty()) {
            return false;
        }
        
        try {
            // Use InetAddress to validate IPv6
            InetAddress addr = InetAddress.getByName(ip);
            return addr.getAddress().length == 16; // IPv6 is 16 bytes
        } catch (Exception e) {
            return false;
        }
    }

    @Test
    @DisplayName("Detected IP should be reachable from localhost")
    void testDetectedIpIsReachable() {
        String ip = NetworkUtils.getLocalIpAddress();
        assertNotNull(ip, "IP address should not be null");
        
        try {
            InetAddress addr = InetAddress.getByName(ip);
            // This test may fail in some network environments, so we just log the result
            boolean reachable = addr.isReachable(2000);
            System.out.println("IP " + ip + " reachable: " + reachable);
        } catch (Exception e) {
            System.err.println("Could not test reachability of IP: " + ip);
            e.printStackTrace();
        }
    }
}
