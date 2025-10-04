package com.sagin.satellite.service.implement;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sagin.satellite.common.SatelliteException;
import com.sagin.satellite.model.Packet;
import com.sagin.satellite.service.IBufferManager;
import com.sagin.satellite.service.ISatelliteService;
import com.sagin.satellite.service.IPacketForwardingService;
import com.sagin.satellite.service.ISatelliteStatusService;

public class SatelliteService implements ISatelliteService {

    private final IBufferManager bufferManager;
    private final IPacketForwardingService packetForwardingService;
    private final ISatelliteStatusService statusService;
    private static final Logger logger = LoggerFactory.getLogger(SatelliteService.class);
    
    public SatelliteService(IBufferManager bufferManager,
                           IPacketForwardingService packetForwardingService,
                           ISatelliteStatusService statusService) {
        this.bufferManager = bufferManager;
        this.packetForwardingService = packetForwardingService;
        this.statusService = statusService;
    }

    @Override
    public void recievePacket(Packet packet) throws Exception {
        logger.info("Satellite received packet: {}", packet.getPacketId());
        
        validatePacket(packet);
        
        try {
            // Kiểm tra xem có phải đích cuối cùng không
            if (packetForwardingService.isDestinationReached(packet, getCurrentNodeId())) {
                // Delivered to final destination
                packetForwardingService.deliverPacket(packet);
                statusService.incrementPacketsProcessed();
                logger.info("Packet {} delivered to final destination", packet.getPacketId());
                return;
            }
            
            // Kiểm tra buffer capacity
            if (!bufferManager.hasCapacity()) {
                logger.warn("Buffer full, dropping packet: {}", packet.getPacketId());
                packetForwardingService.handleDroppedPacket(packet, "Buffer full");
                statusService.incrementPacketsDropped();
                throw new SatelliteException.InvalidPacketException("Buffer full, packet dropped");
            }
            
            // Thêm vào buffer để xử lý
            bufferManager.add(packet);
            logger.info("Packet {} added to buffer for forwarding", packet.getPacketId());
            
            // Cập nhật buffer status
            statusService.updateBufferStatus(bufferManager.getAll());
            
            // Xử lý forwarding ngay lập tức nếu có thể
            processForwarding();
            
        } catch (SatelliteException.InvalidPacketException e) {
            logger.error("Invalid packet {}: {}", packet.getPacketId(), e.getMessage());
            statusService.incrementPacketsDropped();
            throw e;
        } catch (Exception e) {
            logger.error("Error processing packet {}: {}", packet.getPacketId(), e.getMessage());
            statusService.incrementPacketsDropped();
            throw new SatelliteException.InvalidPacketException("Processing error: " + e.getMessage());
        }
    }

    /**
     * Xử lý forwarding các packet trong buffer
     */
    public void processForwarding() {
        logger.debug("Processing packet forwarding...");
        
        while (bufferManager.size() > 0) {
            try {
                Packet packet = bufferManager.poll();
                
                if (packet.isAlive()) {
                    packetForwardingService.forwardPacket(packet);
                    statusService.incrementPacketsProcessed();
                } else {
                    logger.warn("Dead packet found in buffer: {}", packet.getPacketId());
                    packetForwardingService.handleDroppedPacket(packet, "Dead packet");
                    statusService.incrementPacketsDropped();
                }
                
            } catch (SatelliteException.BufferEmptyException e) {
                // Buffer empty, break
                break;
            } catch (Exception e) {
                logger.error("Error during packet forwarding: {}", e.getMessage());
                statusService.incrementPacketsDropped();
            }
        }
        
        // Cập nhật buffer status sau khi processing
        statusService.updateBufferStatus(bufferManager.getAll());
    }

    private void validatePacket(Packet packet) throws Exception {
        if (packet.getPacketId() == null || packet.getPacketId().isEmpty()) {
            throw new SatelliteException.InvalidPacketException("Invalid Packet: Packet ID is missing");
        }
        if (packet.getDestinationUserId() == null || packet.getDestinationUserId().isEmpty()) {
            throw new SatelliteException.InvalidPacketException("Invalid Packet: Destination user ID is missing");
        }
        if (!packet.isAlive()) {
            throw new SatelliteException.InvalidPacketException("Invalid Packet: Packet is not alive");
        }
    }
    
    /**
     * Lấy ID của node hiện tại
     * Trong thực tế, đây sẽ được cấu hình từ properties
     */
    private String getCurrentNodeId() {
        // TODO: Load from configuration
        return System.getProperty("satellite.node.id", "SAT_001");
    }
}
