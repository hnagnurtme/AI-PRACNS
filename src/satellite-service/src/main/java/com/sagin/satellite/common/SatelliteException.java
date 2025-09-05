package com.sagin.satellite.common;


public abstract class SatelliteException {

    public static class BufferEmptyException extends Exception {
        public BufferEmptyException(String message) {
            super(message);
        }
    }

    public static class PacketDroppedException extends Exception {
        public PacketDroppedException(String message) {
            super(message);
        }
    }

    public static class InvalidPacketException extends Exception {
        public InvalidPacketException(String message) {
            super(message);
        }
    }
}
