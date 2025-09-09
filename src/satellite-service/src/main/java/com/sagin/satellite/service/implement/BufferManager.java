package com.sagin.satellite.service.implement;

import com.sagin.satellite.model.Packet;
import com.sagin.satellite.service.IBufferManager;
import com.sagin.satellite.common.SatelliteException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * BufferManager quản lý buffer packet cho SatelliteService
 * Thread-safe, FIFO, maxCapacity, auto-flush và retry.
 * Cải tiến: back-pressure + flush song song nhiều thread + metrics + graceful shutdown
 */
public class BufferManager implements IBufferManager {

    private static final Logger logger = LoggerFactory.getLogger(BufferManager.class);

    // Core components
    private final BlockingQueue<Packet> queue;
    private final int maxCapacity;
    private final TcpSender sender;
    private final ScheduledExecutorService scheduler;
    private final ExecutorService flushPool;
    
    // Configuration
    private final int flushIntervalMs;
    private final int maxRetry;
    private final int backPressureTimeoutMs;
    
    // State management
    private final AtomicBoolean isShuttingDown = new AtomicBoolean(false);
    private final AtomicBoolean isRunning = new AtomicBoolean(true);
    
    // Metrics
    private final AtomicLong packetsAdded = new AtomicLong(0);
    private final AtomicLong packetsDropped = new AtomicLong(0);
    private final AtomicLong packetsSent = new AtomicLong(0);
    private final AtomicLong packetsRetried = new AtomicLong(0);

    /**
     * Constructor với các tham số tối ưu cho hệ thống vệ tinh
     */
    public BufferManager(int maxCapacity, TcpSender sender, int flushIntervalMs, 
                        int maxRetry, int flushThreads) {
        this(maxCapacity, sender, flushIntervalMs, maxRetry, flushThreads, 50);
    }

    /**
     * Constructor đầy đủ với back-pressure timeout
     */
    public BufferManager(int maxCapacity, TcpSender sender, int flushIntervalMs, 
                        int maxRetry, int flushThreads, int backPressureTimeoutMs) {
        validateParameters(maxCapacity, sender, flushIntervalMs, maxRetry, flushThreads, backPressureTimeoutMs);
        
        this.maxCapacity = maxCapacity;
        this.sender = sender;
        this.flushIntervalMs = flushIntervalMs;
        this.maxRetry = maxRetry;
        this.backPressureTimeoutMs = backPressureTimeoutMs;
        
        // Thread-safe queue với capacity limit
        this.queue = new LinkedBlockingQueue<>(this.maxCapacity);
        
        // Scheduler với thread name rõ ràng
        this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "BufferManager-Scheduler");
            t.setDaemon(true);
            return t;
        });
        
        // Flush thread pool với thread name
        this.flushPool = Executors.newFixedThreadPool(flushThreads, r -> {
            Thread t = new Thread(r, "BufferManager-Flush-" + System.currentTimeMillis());
            t.setDaemon(true);
            return t;
        });

        // Khởi động auto flush
        startAutoFlush();
        
        logger.info("BufferManager initialized: capacity={}, flushInterval={}ms, maxRetry={}, flushThreads={}", 
                   maxCapacity, flushIntervalMs, maxRetry, flushThreads);
    }

    private void validateParameters(int maxCapacity, TcpSender sender, int flushIntervalMs, 
                                  int maxRetry, int flushThreads, int backPressureTimeoutMs) {
        if (maxCapacity <= 0) {
            throw new IllegalArgumentException("maxCapacity must be positive");
        }
        if (sender == null) {
            throw new IllegalArgumentException("sender cannot be null");
        }
        if (flushIntervalMs <= 0) {
            throw new IllegalArgumentException("flushIntervalMs must be positive");
        }
        if (maxRetry < 0) {
            throw new IllegalArgumentException("maxRetry cannot be negative");
        }
        if (flushThreads <= 0) {
            throw new IllegalArgumentException("flushThreads must be positive");
        }
        if (backPressureTimeoutMs < 0) {
            throw new IllegalArgumentException("backPressureTimeoutMs cannot be negative");
        }
    }

    private void startAutoFlush() {
        scheduler.scheduleAtFixedRate(() -> {
            if (!isShuttingDown.get()) {
                try {
                    flush();
                } catch (Exception e) {
                    logger.error("Error during auto flush", e);
                }
            }
        }, flushIntervalMs, flushIntervalMs, TimeUnit.MILLISECONDS);
    }

    @Override
    public void add(Packet packet) throws SatelliteException.InvalidPacketException {
        if (isShuttingDown.get()) {
            throw new SatelliteException.InvalidPacketException("BufferManager is shutting down");
        }
        
        if (packet == null) {
            packetsDropped.incrementAndGet();
            throw new SatelliteException.InvalidPacketException("Packet cannot be null");
        }
        
        if (packet.getPacketId() == null || packet.getPacketId().trim().isEmpty()) {
            packetsDropped.incrementAndGet();
            throw new SatelliteException.InvalidPacketException("Packet ID cannot be null or empty");
        }

        try {
            boolean added = queue.offer(packet, backPressureTimeoutMs, TimeUnit.MILLISECONDS);
            if (added) {
                packetsAdded.incrementAndGet();
                logger.debug("Packet {} added to buffer. Queue size: {}/{}", 
                           packet.getPacketId(), queue.size(), maxCapacity);
            } else {
                packetsDropped.incrementAndGet();
                logger.warn("Buffer full after {}ms timeout, dropping packet {}. Queue size: {}/{}", 
                          backPressureTimeoutMs, packet.getPacketId(), queue.size(), maxCapacity);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            packetsDropped.incrementAndGet();
            logger.error("Interrupted while adding packet {}", packet.getPacketId());
            throw new SatelliteException.InvalidPacketException("Interrupted while adding packet");
        }
    }

    @Override
    public Packet poll() throws SatelliteException.BufferEmptyException {
        if (isShuttingDown.get()) {
            throw new SatelliteException.BufferEmptyException("BufferManager is shutting down");
        }
        
        Packet packet = queue.poll();
        if (packet == null) {
            throw new SatelliteException.BufferEmptyException("Buffer is empty");
        }
        
        logger.debug("Packet {} polled from buffer. Remaining: {}", packet.getPacketId(), queue.size());
        return packet;
    }

    @Override
    public List<Packet> getAll() {
        return new ArrayList<>(queue);
    }

    @Override
    public boolean hasCapacity() {
        return queue.remainingCapacity() > 0 && !isShuttingDown.get();
    }

    @Override
    public int size() {
        return queue.size();
    }

    @Override
    public void clear() {
        int clearedSize = queue.size();
        queue.clear();
        logger.info("Buffer cleared, {} packets removed", clearedSize);
    }

    public void flush() {
        if (isShuttingDown.get()) {
            return;
        }
        
        int flushedCount = 0;
        Packet packet;
        
        while ((packet = queue.poll()) != null) {
            final Packet finalPacket = packet;
            flushedCount++;
            
            flushPool.execute(() -> {
                if (!isShuttingDown.get()) {
                    sendWithRetry(finalPacket);
                }
            });
        }
        
        if (flushedCount > 0) {
            logger.debug("Flushed {} packets from buffer", flushedCount);
        }
    }

    private void sendWithRetry(Packet packet) {
        int retries = 0;
        boolean sent = false;
        
        while (!sent && retries <= maxRetry && !isShuttingDown.get()) {
            try {
                sender.send(packet);
                sent = true;
                packetsSent.incrementAndGet();
                logger.debug("Packet {} sent successfully on attempt {}", packet.getPacketId(), retries + 1);
            } catch (Exception ex) {
                retries++;
                packetsRetried.incrementAndGet();
                
                if (retries <= maxRetry) {
                    logger.warn("Failed to send packet {}, retry {}/{}: {}", 
                              packet.getPacketId(), retries, maxRetry, ex.getMessage());
                    
                    try {
                        Thread.sleep(Math.min(1000, 100 * retries));
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        logger.error("Interrupted during retry backoff for packet {}", packet.getPacketId());
                        break;
                    }
                } else {
                    packetsDropped.incrementAndGet();
                    logger.error("Packet {} dropped after {} retries: {}", 
                               packet.getPacketId(), maxRetry, ex.getMessage());
                }
            }
        }
    }

    public void shutdown() {
        shutdown(flushIntervalMs * 3);
    }

    /**
     * Graceful shutdown với timeout tùy chỉnh
     */
    public void shutdown(long timeoutMs) {
        if (isShuttingDown.compareAndSet(false, true)) {
            logger.info("BufferManager shutdown initiated. Flushing remaining {} packets...", queue.size());
            
            flush();
            
            scheduler.shutdown();
            
            flushPool.shutdown();
            
            try {
                if (!scheduler.awaitTermination(timeoutMs / 3, TimeUnit.MILLISECONDS)) {
                    logger.warn("Scheduler didn't terminate gracefully, forcing shutdown");
                    scheduler.shutdownNow();
                }
                
                if (!flushPool.awaitTermination(timeoutMs * 2 / 3, TimeUnit.MILLISECONDS)) {
                    logger.warn("Flush pool didn't terminate gracefully, forcing shutdown");
                    flushPool.shutdownNow();
                }
                
                isRunning.set(false);
                logger.info("BufferManager shutdown completed. Final stats: {}", getMetrics());
                
            } catch (InterruptedException e) {
                logger.error("Interrupted during shutdown, forcing immediate termination");
                scheduler.shutdownNow();
                flushPool.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    public boolean isRunning() {
        return isRunning.get() && !isShuttingDown.get();
    }

    public BufferMetrics getMetrics() {
        return new BufferMetrics(
            packetsAdded.get(),
            packetsSent.get(),
            packetsDropped.get(),
            packetsRetried.get(),
            queue.size(),
            maxCapacity,
            queue.remainingCapacity()
        );
    }

    /**
     * Health check cho buffer
     */
    public boolean isHealthy() {
        return isRunning() && 
               !scheduler.isShutdown() && 
               !flushPool.isShutdown() &&
               sender != null;
    }

    public static class BufferMetrics {
        private final long packetsAdded;
        private final long packetsSent;
        private final long packetsDropped;
        private final long packetsRetried;
        private final int currentSize;
        private final int maxCapacity;
        private final int remainingCapacity;

        public BufferMetrics(long packetsAdded, long packetsSent, long packetsDropped, 
                           long packetsRetried, int currentSize, int maxCapacity, int remainingCapacity) {
            this.packetsAdded = packetsAdded;
            this.packetsSent = packetsSent;
            this.packetsDropped = packetsDropped;
            this.packetsRetried = packetsRetried;
            this.currentSize = currentSize;
            this.maxCapacity = maxCapacity;
            this.remainingCapacity = remainingCapacity;
        }

        // Getters
        public long getPacketsAdded() { return packetsAdded; }
        public long getPacketsSent() { return packetsSent; }
        public long getPacketsDropped() { return packetsDropped; }
        public long getPacketsRetried() { return packetsRetried; }
        public int getCurrentSize() { return currentSize; }
        public int getMaxCapacity() { return maxCapacity; }
        public int getRemainingCapacity() { return remainingCapacity; }
        
        public double getDropRate() {
            return packetsAdded > 0 ? (double) packetsDropped / packetsAdded : 0.0;
        }
        
        public double getUtilizationRate() {
            return (double) currentSize / maxCapacity;
        }

        @Override
        public String toString() {
            return String.format("BufferMetrics{added=%d, sent=%d, dropped=%d, retried=%d, " +
                               "size=%d/%d, dropRate=%.2f%%, utilization=%.2f%%}", 
                               packetsAdded, packetsSent, packetsDropped, packetsRetried,
                               currentSize, maxCapacity, getDropRate() * 100, getUtilizationRate() * 100);
        }
    }
}