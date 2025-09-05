package com.sagin.satellite.service.implement;



import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.sagin.satellite.common.SatelliteException;
import com.sagin.satellite.model.Packet;
import com.sagin.satellite.service.IBufferManager;
import com.sagin.satellite.service.ISatelliteService;

public class SatelliteService implements ISatelliteService {

    private final IBufferManager bufferManager;
    private static final Logger logger = LoggerFactory.getLogger(SatelliteService.class);
    public SatelliteService(IBufferManager bufferManager) {
        this.bufferManager = bufferManager;
    }
    /**    
     * Add a packet to the buffer for processing.
     * @param packet The packet to be added to the buffer.
     * Writes log messages for debugging and error handling.
     * Update sattelite status 
     */
    @Override
    public void recievePacket(Packet packet) throws Exception {
        validatePacket(packet);
        bufferManager.add(packet);

        logger.info("Packet {} added to buffer", packet.getPacketId());
        
    }

    private void validatePacket(Packet packet) throws Exception {
        if (packet.getPacketId() == null || packet.getPacketId().isEmpty()) {
            throw new SatelliteException.InvalidPacketException("Invalid Packet: Packet ID is missing");
        }
        if( packet.getNextHop() == null || packet.getNextHop().isEmpty()){
            throw new SatelliteException.InvalidPacketException("Invalid Packet: Next hop is missing");
        }
    }
}


