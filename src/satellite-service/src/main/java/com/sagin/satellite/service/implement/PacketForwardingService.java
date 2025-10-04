package com.sagin.satellite.service.implement;

import com.sagin.satellite.model.Packet;
import com.sagin.satellite.service.IPacketForwardingService;
import com.sagin.satellite.service.IRoutingService;
import com.sagin.satellite.service.ITcpSender;
import com.sagin.satellite.common.SatelliteException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * PacketForwardingService xử lý logic chuyển tiếp packet
 */
public class PacketForwardingService implements IPacketForwardingService {

    private static final Logger logger = LoggerFactory.getLogger(PacketForwardingService.class);
    
    private final IRoutingService routingService;
    private final ITcpSender tcpSender;
    private final String currentNodeId;

    public PacketForwardingService(IRoutingService routingService, 
                                 ITcpSender tcpSender, 
                                 String currentNodeId) {
        this.routingService = routingService;
        this.tcpSender = tcpSender;
        this.currentNodeId = currentNodeId;
    }

    @Override
    public void forwardPacket(Packet packet) throws SatelliteException.SendException, SatelliteException.InvalidPacketException {
        logger.debug("Forwarding packet {} from node {}", packet.getPacketId(), currentNodeId);
        
        // Validate packet
        validatePacketForForwarding(packet);
        
        // Check TTL
        if (packet.getTTL() <= 0) {
            handleDroppedPacket(packet, "TTL expired");
            throw new SatelliteException.InvalidPacketException("Packet TTL expired");
        }
        
        // Decrement TTL
        packet.setTTL(packet.getTTL() - 1);
        
        // Add current node to path history
        packet.addToPath(currentNodeId);
        packet.setCurrentNode(currentNodeId);
        
        // Check if this is the final destination
        if (isDestinationReached(packet, currentNodeId)) {
            deliverPacket(packet);
            return;
        }
        
        // Update routing information
        if (!updatePacketRouting(packet, currentNodeId)) {
            handleDroppedPacket(packet, "No route to destination");
            throw new SatelliteException.SendException("No route found to destination: " + packet.getDestinationUserId());
        }
        
        // Forward to next hop
        try {
            tcpSender.send(packet);
            logger.info("Packet {} forwarded to next hop: {}", packet.getPacketId(), packet.getNextHop());
        } catch (Exception e) {
            logger.error("Failed to forward packet {}: {}", packet.getPacketId(), e.getMessage());
            
            // Retry logic
            packet.incrementRetry();
            if (packet.getRetryCount() < 3) {
                logger.info("Retrying packet {} (attempt {})", packet.getPacketId(), packet.getRetryCount());
                // Could implement exponential backoff here
                throw new SatelliteException.SendException("Temporary send failure, will retry", e);
            } else {
                handleDroppedPacket(packet, "Max retries exceeded");
                throw new SatelliteException.SendException("Max retries exceeded for packet: " + packet.getPacketId(), e);
            }
        }
    }

    @Override
    public void deliverPacket(Packet packet) {
        logger.info("Packet {} delivered to final destination", packet.getPacketId());
        
        // Update delivery metrics
        long currentTime = System.currentTimeMillis();
        double totalDelay = currentTime - packet.getTimestamp();
        packet.setDelayMs(totalDelay);
        
        // Log delivery statistics
        logger.info("Packet {} delivery stats: delay={}ms, path={}, retries={}", 
                   packet.getPacketId(), totalDelay, packet.getPathHistory(), packet.getRetryCount());
        
        // Here you could:
        // 1. Send to final destination application
        // 2. Store delivery confirmation
        // 3. Update network statistics
        // 4. Notify sender about successful delivery
    }

    @Override
    public boolean isDestinationReached(Packet packet, String currentNodeId) {
        // For now, we'll use a simple mapping where destination user is served by a specific node
        // In a real system, this would involve user location lookup
        String destinationNodeId = getUserServingNode(packet.getDestinationUserId());
        return currentNodeId.equals(destinationNodeId);
    }

    @Override
    public boolean updatePacketRouting(Packet packet, String currentNodeId) {
        // Get destination node for the target user
        String destinationNodeId = getUserServingNode(packet.getDestinationUserId());
        
        if (destinationNodeId == null) {
            logger.warn("No serving node found for user: {}", packet.getDestinationUserId());
            return false;
        }
        
        // Find next hop using routing service
        String nextHop = routingService.findNextHop(currentNodeId, destinationNodeId);
        
        if (nextHop == null) {
            logger.warn("No route found from {} to {}", currentNodeId, destinationNodeId);
            return false;
        }
        
        packet.setNextHop(nextHop);
        return true;
    }

    @Override
    public void handleDroppedPacket(Packet packet, String reason) {
        logger.warn("Dropping packet {}: {}", packet.getPacketId(), reason);
        
        packet.markDropped();
        
        // Update metrics
        long currentTime = System.currentTimeMillis();
        double totalDelay = currentTime - packet.getTimestamp();
        packet.setDelayMs(totalDelay);
        
        // Log dropped packet statistics
        logger.info("Dropped packet {} stats: reason={}, delay={}ms, path={}, retries={}", 
                   packet.getPacketId(), reason, totalDelay, packet.getPathHistory(), packet.getRetryCount());
        
        // Here you could:
        // 1. Send notification to source about packet drop
        // 2. Update network failure statistics
        // 3. Trigger network topology updates if needed
    }

    /**
     * Validate packet before forwarding
     */
    private void validatePacketForForwarding(Packet packet) throws SatelliteException.InvalidPacketException {
        if (packet == null) {
            throw new SatelliteException.InvalidPacketException("Packet is null");
        }
        
        if (packet.getPacketId() == null || packet.getPacketId().isEmpty()) {
            throw new SatelliteException.InvalidPacketException("Packet ID is missing");
        }
        
        if (packet.getDestinationUserId() == null || packet.getDestinationUserId().isEmpty()) {
            throw new SatelliteException.InvalidPacketException("Destination user ID is missing");
        }
        
        if (packet.isDropped()) {
            throw new SatelliteException.InvalidPacketException("Packet is already dropped");
        }
        
        if (!packet.isAlive()) {
            throw new SatelliteException.InvalidPacketException("Packet is not alive");
        }
    }

    /**
     * Get the node that serves a specific user
     * This is a simplified implementation - in reality, this would involve
     * user location services, load balancing, etc.
     */
    private String getUserServingNode(String userId) {
        // Simple hash-based distribution for demo
        // In a real system, this would query a user location service
        if (userId == null) return null;
        
        int hash = Math.abs(userId.hashCode());
        int nodeIndex = hash % 3; // Assume 3 nodes: SAT_001, SAT_002, SAT_003
        
        return "SAT_" + String.format("%03d", nodeIndex + 1);
    }
}