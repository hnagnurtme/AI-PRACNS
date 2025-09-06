package com.sagin.satellite.service.implement;

import com.sagin.satellite.model.Packet;
import com.sagin.satellite.service.IBufferManager;
import com.sagin.satellite.common.SatelliteException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * BufferManager quản lý buffer packet cho SatelliteService
 * Thread-safe, FIFO, maxCapacity, auto-flush và retry.
 * Cải tiến: back-pressure + flush song song nhiều thread
 */
public class BufferManager implements IBufferManager {

    private static final Logger logger = LoggerFactory.getLogger(BufferManager.class);

    private final BlockingQueue<Packet> queue;
    private final int maxCapacity;
    private final TcpSender sender;
    private final ScheduledExecutorService scheduler;
    private final ExecutorService flushPool;
    private final int flushIntervalMs;
    private final int maxRetry;

    public BufferManager(int maxCapacity, TcpSender sender, int flushIntervalMs, int maxRetry, int flushThreads) {
        this.maxCapacity = maxCapacity;
        this.sender = sender;
        this.flushIntervalMs = flushIntervalMs;
        this.maxRetry = maxRetry;
        this.queue = new LinkedBlockingQueue<>(this.maxCapacity);
        this.scheduler = Executors.newSingleThreadScheduledExecutor();
        this.flushPool = Executors.newFixedThreadPool(flushThreads);

        // Auto flush queue mỗi flushIntervalMs
        scheduler.scheduleAtFixedRate(this::flush, flushIntervalMs, flushIntervalMs, TimeUnit.MILLISECONDS);
    }

    @Override
    public void add(Packet packet) throws SatelliteException.InvalidPacketException {
        if (packet == null || packet.getPacketId() == null) {
            throw new SatelliteException.InvalidPacketException("Packet is null or missing ID");
        }

        try {
            // Chờ tối đa 50ms nếu buffer đầy (back-pressure)
            boolean added = queue.offer(packet, 50, TimeUnit.MILLISECONDS);
            if (!added) {
                logger.warn("Buffer full — waiting 50ms still full, dropping packet {}", packet.getPacketId());
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.error("Interrupted while waiting to add packet {}", packet.getPacketId());
        }
    }

    @Override
    public Packet poll() throws SatelliteException.BufferEmptyException {
        Packet packet = queue.poll();
        if (packet == null) {
            throw new SatelliteException.BufferEmptyException("Buffer is empty");
        }
        return packet;
    }

    @Override
    public List<Packet> getAll() {
        return queue.stream().collect(Collectors.toList());
    }

    @Override
    public boolean hasCapacity() {
        return queue.remainingCapacity() > 0;
    }

    @Override
    public int size() {
        return queue.size();
    }

    @Override
    public void clear() {
        queue.clear();
    }

    private void flush() {
        Packet packet;
        while ((packet = queue.poll()) != null) {
            Packet finalPacket = packet;
            flushPool.execute(() -> sendWithRetry(finalPacket));
        }
    }

    private void sendWithRetry(Packet packet) {
        int retries = 0;
        boolean sent = false;
        while (!sent && retries <= maxRetry) {
            try {
                sender.send(packet);
                sent = true;
                logger.info("Packet {} sent successfully", packet.getPacketId());
            } catch (Exception ex) {
                retries++;
                logger.error("Failed to send packet {}, retry {}/{}", packet.getPacketId(), retries, maxRetry, ex);
                if (retries > maxRetry) {
                    logger.error("Packet {} dropped after {} retries", packet.getPacketId(), maxRetry);
                }
            }
        }
    }

    public void shutdown() {
        scheduler.shutdown();
        flushPool.shutdown();
        try {
            if (!scheduler.awaitTermination(flushIntervalMs, TimeUnit.MILLISECONDS)) {
                scheduler.shutdownNow();
            }
            if (!flushPool.awaitTermination(flushIntervalMs * 2, TimeUnit.MILLISECONDS)) {
                flushPool.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
            flushPool.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    public boolean isFull() {
        return !hasCapacity();
    }
}
