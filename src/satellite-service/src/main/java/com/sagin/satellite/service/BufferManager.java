package com.sagin.satellite.service;

import com.sagin.satellite.model.Packet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Queue;
import java.util.concurrent.*;

public class BufferManager {
    private static final Logger logger = LoggerFactory.getLogger(BufferManager.class);
    private final Queue<Packet> queue = new ConcurrentLinkedQueue<>();
    private final int maxSize;
    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
    private final TcpSender sender;

    public BufferManager(int maxSize, TcpSender sender) {
        this.maxSize = maxSize;
        this.sender = sender;
        scheduler.scheduleAtFixedRate(this::flush, 200, 200, TimeUnit.MILLISECONDS);
    }

    public boolean shouldBuffer() {
        return queue.size() >= maxSize;
    }

    public boolean enqueue(Packet packet) {
        if (queue.size() >= maxSize) {
            logger.warn("Buffer full — dropping packet {}", packet.getId());
            return false;
        }
        queue.offer(packet);
        return true;
    }

    private void flush() {
        try {
            Packet p;
            while ((p = queue.poll()) != null) {
                try {
                    sender.send(p);
                } catch (Exception ex) {
                    logger.error("Failed to send buffered packet {}, requeueing", p.getId(), ex);
                    queue.offer(p);
                    break;
                }
            }
        } catch (Exception e) {
            logger.error("Buffer flush error", e);
        }
    }

    public void add(Packet packet) {
        if (!enqueue(packet)) {
            logger.warn("Dropping packet {}", packet.getId());
        }
    }

    public Packet poll() {
        return queue.poll();
    }

    public void shutdown() {
        scheduler.shutdownNow();
    }
}
