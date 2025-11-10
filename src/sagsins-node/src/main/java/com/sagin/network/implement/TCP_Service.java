package com.sagin.network.implement;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sagin.DTOs.RoutingRequest;
import com.sagin.helper.PacketHelper;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.model.UserInfo;
import com.sagin.network.interfaces.ITCP_Service;
import com.sagin.repository.INodeRepository;
import com.sagin.repository.IUserRepository;
import com.sagin.service.INodeService;
import com.sagin.routing.IRoutingService;
import com.sagin.routing.RLRoutingService;
import com.sagin.routing.RouteInfo;
import com.sagin.util.SimulationConstants;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.Optional;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * TCP Service implementation with support for scheduled RETRIES and complex ROUTING logic.
 *
 * Maintains a clear separation of concerns for Receive (RX) and Transmit (TX)
 * resource accounting.
 * * - The "receivePacket" (Producer) is fast, non-blocking, and enqueues packets.
 * - The "processSendQueue" (Consumer) is an asynchronous, single-threaded
 * process that handles blocking I/O, retries, and simulation accounting.
 */
public class TCP_Service implements ITCP_Service {
    private static final Logger logger = LoggerFactory.getLogger(TCP_Service.class);

    // --- Dependencies ---
    private final INodeRepository nodeRepository;
    private final IUserRepository userRepository;
    private final INodeService nodeService;
    private final IRoutingService routingService;
    private final ObjectMapper objectMapper;
    private final com.sagin.service.BatchPacketService batchPacketService;
    private final RLRoutingService rlRoutingService; // Injected dependency

    // --- Retry Queue & Scheduler ---
    private final BlockingQueue<RetryablePacket> sendQueue;
    private final ScheduledExecutorService retryScheduler;
    private static final int MAX_RETRIES = 3;
    private static final long RETRY_POLL_INTERVAL_MS = 50; // Poll interval for the send queue

    /**
     * Internal record to pass state from the RX path (producer) to the TX
     * path (consumer).
     * This context is used to create a HopRecord with actual delays AFTER a
     * successful send.
     */
    private record HopContext(
            NodeInfo currentNode,
            NodeInfo nextNode,
            RouteInfo routeInfo,
            double rxCpuDelay) { // The RX/CPU cost calculated at the time of receipt
    }

    /**
     * Internal object to encapsulate a packet, its destination, and retry state.
     * The 'originalNodeId' is crucial for debiting TX resources from the
     * correct node after a successful send.
     */
    private record RetryablePacket(
            String originalNodeId, // The node *sending* the packet (for TX accounting)
            Packet packet,
            String host,
            int port,
            String destinationDesc, // For logging (e.g., "NODE:xyz" or "USER:abc")
            int attemptCount,
            HopContext hopContext) { // The state captured during RX
    }

    /**
     * Initializes the TCP_Service with all dependencies.
     */
    public TCP_Service(INodeRepository nodeRepository,
            INodeService nodeService,
            IUserRepository userRepository,
            IRoutingService routingService,
            com.sagin.service.BatchPacketService batchPacketService,
            RLRoutingService rlRoutingService // ✅ OPTIMIZATION 1: Injected Dependency
    ) {
        this.nodeRepository = nodeRepository;
        this.nodeService = nodeService;
        this.userRepository = userRepository;
        this.routingService = routingService;
        this.batchPacketService = batchPacketService;
        this.rlRoutingService = rlRoutingService; // Assign injected dependency

        // Initialize ObjectMapper and register JavaTimeModule (for Instant, etc.)
        this.objectMapper = new ObjectMapper().registerModule(new JavaTimeModule());

        // Initialize queue and background service
        this.sendQueue = new LinkedBlockingQueue<>();
        this.retryScheduler = Executors.newSingleThreadScheduledExecutor();

        // Start the background queue processing thread
        this.startSendScheduler();
    }

    /**
     * Entry point for a packet received by this node (from NodeGateway).
     * This is the "Producer" side: fast, non-blocking.
     *
     * @param packet The packet received from the network.
     */
    @Override
    public void receivePacket(Packet packet) {
        if (packet == null || packet.getCurrentHoldingNodeId() == null) {
            logger.warn("[TCP_Service] Received an invalid or null packet.");
            return;
        }

        String currentNodeId = packet.getCurrentHoldingNodeId();
        logger.info("[TCP_Service] 📥 Nhận Packet {} tại {} | TTL: {} | Delay hiện tại: {}ms",
                packet.getPacketId(), currentNodeId, packet.getTTL(),
                String.format("%.2f", packet.getAccumulatedDelayMs()));

        // === STEP 1: Account for RECEIVE (RX/CPU) costs ===
        double rxCpuDelay = 0.0;
        try {
            rxCpuDelay = nodeService.updateNodeStatus(currentNodeId, packet);
        } catch (Exception e) {
            logger.error("[TCP_Service] Error during RX/CPU accounting for {}: {}", currentNodeId, e.getMessage(), e);
        }

        // Check if the packet was dropped during updateNodeStatus (e.g., queue
        // overflow)
        if (packet.isDropped()) {
            logger.warn("[TCP_Service] Packet {} dropped at {}: {}",
                    packet.getPacketId(), currentNodeId, packet.getDropReason());
            return;
        }

        // === STEP 2: Check destination ===
        if (currentNodeId.equals(packet.getStationDest())) {
            // ✅ Packet has arrived at the destination station.
            // Prepare for the final hop (Station -> User).

            // Decrement TTL (this is also a hop)
            packet.setTTL(packet.getTTL() - 1);
            if (packet.getTTL() <= 0) {
                packet.setDropped(true);
                packet.setDropReason("TTL_EXPIRED");
                logger.warn("[TCP_Service] Packet {} dropped: TTL expired at destination station {}.",
                        packet.getPacketId(), currentNodeId);
                return;
            }

            // Update path history
            if (packet.getPathHistory() != null && !packet.getPathHistory().contains(currentNodeId)) {
                packet.getPathHistory().add(currentNodeId);
            }

            logger.info("[TCP_Service] ✅ Packet {} reached destination station {} | TTL: {} | Forwarding to user...",
                    packet.getPacketId(), currentNodeId, packet.getTTL());

            // Forward to user WITH context to create the final HopRecord
            forwardPacketToUserWithContext(packet, currentNodeId, rxCpuDelay);
            return;
        }

        // === STEP 3: Routing (Transit Node) ===
        RouteInfo bestRoute = getBestRoute(packet);
        if (bestRoute == null) {
            packet.setDropped(true);
            packet.setDropReason("NO_ROUTE_TO_HOST");
            logger.warn("[TCP_Service] No route found for packet {} from {} to {}.",
                    packet.getPacketId(), currentNodeId, packet.getStationDest());
            return;
        }

        String nextHopNodeId = bestRoute.getNextHopNodeId();
        packet.setNextHopNodeId(nextHopNodeId);

        Optional<NodeInfo> currentNodeOpt = nodeRepository.getNodeInfo(currentNodeId);
        Optional<NodeInfo> nextNodeOpt = nodeRepository.getNodeInfo(nextHopNodeId);

        if (currentNodeOpt.isEmpty() || nextNodeOpt.isEmpty()) {
            packet.setDropped(true);
            packet.setDropReason("NODE_INFO_NOT_FOUND");
            logger.error("[TCP_Service] Could not find node info for {} or {}.", currentNodeId, nextHopNodeId);
            return;
        }

        NodeInfo currentNode = currentNodeOpt.get();
        NodeInfo nextNode = nextNodeOpt.get();

        // === STEP 4: Prepare packet for transit (TTL, pathHistory) ===
        PacketHelper.preparePacketForTransit(packet, nextNode);

        if (packet.isDropped()) {
            logger.warn("[TCP_Service] Packet {} dropped: {}", packet.getPacketId(), packet.getDropReason());
            return;
        }

        logger.info("[TCP_Service] 🔄 Routing Packet {} | {} → {} (next: {}) | Delay: {}ms",
                packet.getPacketId(), currentNodeId, packet.getStationDest(), nextHopNodeId,
                String.format("%.2f", packet.getAccumulatedDelayMs()));

        // === STEP 5: Enqueue for sending (TX) and create HopRecord AFTER send ===
        packet.setCurrentHoldingNodeId(nextHopNodeId);
        sendPacketWithContext(packet, currentNodeId, currentNode, nextNode, bestRoute, rxCpuDelay);
    }

    /**
     * Sends a packet with full context to create a HopRecord after successful
     * transmission.
     */
    private void sendPacketWithContext(Packet packet, String senderNodeId,
            NodeInfo currentNode, NodeInfo nextNode,
            RouteInfo routeInfo, double rxCpuDelay) {
        String nextHopNodeId = packet.getNextHopNodeId();
        String host = nextNode.getCommunication().getIpAddress();
        int port = nextNode.getCommunication().getPort();

        if (host == null || port <= 0) {
            logger.warn("[TCP_Service] Packet {} from {} dropped: Node {} has invalid host/port.",
                    packet.getPacketId(), senderNodeId, nextHopNodeId);
            packet.setDropped(true);
            packet.setDropReason("INVALID_HOST_PORT");
            return;
        }

        // Create the context to pass to the consumer thread
        HopContext context = new HopContext(currentNode, nextNode, routeInfo, rxCpuDelay);
        addToSendQueueWithContext(senderNodeId, packet, host, port, "NODE:" + nextHopNodeId, context);
    }

    /**
     * (Node-to-Node Send) - LEGACY/TEST only.
     * Enqueues a packet for sending to another node.
     *
     * @param packet       The packet to send.
     * @param senderNodeId The CURRENT node sending this packet (for TX accounting).
     */
    @Override
    public void sendPacket(Packet packet, String senderNodeId) {
        String nextHopNodeId = packet.getNextHopNodeId();
        if (nextHopNodeId == null) {
            logger.warn("[TCP_Service] Packet {} from {} dropped: nextHopNodeId is null (Routing black hole).",
                    packet.getPacketId(), senderNodeId);
            packet.setDropped(true);
            packet.setDropReason("ROUTING_BLACK_HOLE");
            return;
        }

        Optional<NodeInfo> nextHopOpt = nodeRepository.getNodeInfo(nextHopNodeId);
        if (nextHopOpt.isEmpty()) {
            logger.warn("[TCP_Service] Packet {} from {} dropped: Node {} not found in DB (Routing error).",
                    packet.getPacketId(), senderNodeId, nextHopNodeId);
            packet.setDropped(true);
            packet.setDropReason("ROUTING_NODE_NOT_FOUND");
            return;
        }
        // Get host/port info for the next hop
        NodeInfo nextHop = nextHopOpt.get();
        String host = nextHop.getCommunication().getIpAddress();
        int port = nextHop.getCommunication().getPort();
        if (host == null || port <= 0) {
            logger.warn("[TCP_Service] Packet {} from {} dropped: Node {} has invalid host/port.",
                    packet.getPacketId(), senderNodeId, nextHopNodeId);
            return;
        }

        // Add to queue, carrying the senderNodeId
        addToSendQueue(senderNodeId, packet, host, port, "NODE:" + nextHopNodeId);
    }

    /**
     * Forwards a packet to the end-user WITH CONTEXT to create the final
     * HopRecord.
     * This is the final hop: Station -> User.
     *
     * @param packet       The packet to send.
     * @param senderNodeId The CURRENT node (destination station) sending the
     * packet.
     * @param rxCpuDelay   The RX/CPU delay already calculated.
     */
    private void forwardPacketToUserWithContext(Packet packet, String senderNodeId, double rxCpuDelay) {
        String userId = packet.getDestinationUserId();
        if (userId == null || userId.isBlank()) {
            logger.warn("[TCP_Service] (forwardUser) Cannot forward {}: destinationUserId is null.",
                    packet.getPacketId());
            return;
        }

        Optional<UserInfo> userOpt = userRepository.findByUserId(userId);
        if (userOpt.isEmpty()) {
            logger.error("[TCP_Service] (forwardUser) User {} not found. Cannot deliver packet {}.", userId,
                    packet.getPacketId());
            return;
        }

        UserInfo user = userOpt.get();
        String host = user.getIpAddress();
        int port = user.getPort();
        if (host == null || port <= 0) {
            logger.error("[TCP_Service] (forwardUser) User {} has invalid host/port info.", userId);
            return;
        }

        // Get current node info to build the HopContext
        Optional<NodeInfo> currentNodeOpt = nodeRepository.getNodeInfo(senderNodeId);
        if (currentNodeOpt.isEmpty()) {
            logger.error("[TCP_Service] (forwardUser) Could not find node info for {}.", senderNodeId);
            // Fallback: Send without context
            addToSendQueue(senderNodeId, packet, host, port, "USER:" + userId);
            return;
        }

        NodeInfo currentNode = currentNodeOpt.get();

        // Create HopContext for the final hop (Station -> User)
        // nextNode = null because the destination is a User, not a Node.
        // routeInfo = null because no routing decision is needed for the final hop.
        HopContext context = new HopContext(currentNode, null, null, rxCpuDelay);

        // Add to queue WITH context
        addToSendQueueWithContext(senderNodeId, packet, host, port, "USER:" + userId, context);
    }

    /**
     * (Node-to-User Send) - LEGACY/FALLBACK
     * Enqueues a packet for sending to a user WITHOUT HopContext.
     */
    @SuppressWarnings("unused")
    private void forwardPacketToUser(Packet packet, String senderNodeId) {
        String userId = packet.getDestinationUserId();
        if (userId == null || userId.isBlank()) {
            logger.warn("[TCP_Service] (forwardUser) Cannot forward {}: destinationUserId is null.",
                    packet.getPacketId());
            return;
        }

        Optional<UserInfo> userOpt = userRepository.findByUserId(userId);
        if (userOpt.isEmpty()) {
            logger.error("[TCP_Service] (forwardUser) User {} not found. Cannot deliver packet {}.", userId,
                    packet.getPacketId());
            return;
        }

        UserInfo user = userOpt.get();
        String host = user.getIpAddress();
        int port = user.getPort();
        if (host == null || port <= 0) {
            logger.error("[TCP_Service] (forwardUser) User {} has invalid host/port info.", userId);
            return;
        }

        // Add to queue, carrying the senderNodeId
        addToSendQueue(senderNodeId, packet, host, port, "USER:" + userId);
    }

    // ===================================================================
    // ASYNC PRODUCER/CONSUMER QUEUE MANAGEMENT
    // ===================================================================

    /**
     * (Producer)
     * Adds a packet to the send queue WITH CONTEXT for HopRecord creation.
     */
    private void addToSendQueueWithContext(String originalNodeId, Packet packet, String host, int port,
            String destinationDesc, HopContext context) {
        RetryablePacket job = new RetryablePacket(originalNodeId, packet, host, port, destinationDesc, 1, context);
        try {
            sendQueue.put(job);
            logger.debug("[TCP_Service] ✈️ Enqueued Packet {} (from {}) for send → {}.",
                    packet.getPacketId(), originalNodeId, destinationDesc);
        } catch (InterruptedException e) {
            logger.error("[TCP_Service] Interrupted while enqueuing packet {}.", packet.getPacketId(), e);
            Thread.currentThread().interrupt();
        }
    }

    /**
     * (Producer) - LEGACY
     * Adds to queue WITHOUT context (e.g., for forwardPacketToUser fallback).
     */
    private void addToSendQueue(String originalNodeId, Packet packet, String host, int port, String destinationDesc) {
        addToSendQueueWithContext(originalNodeId, packet, host, port, destinationDesc, null);
    }

    /**
     * (Consumer Setup)
     * Starts a background thread to process the `sendQueue`.
     */
    private void startSendScheduler() {
        this.retryScheduler.scheduleAtFixedRate(this::processSendQueue,
                RETRY_POLL_INTERVAL_MS,
                RETRY_POLL_INTERVAL_MS,
                TimeUnit.MILLISECONDS);
        logger.info("[TCP_Service] Send Scheduler service started.");
    }

    /**
     * (Consumer Logic)
     * This method is called periodically by the `retryScheduler` to drain the
     * queue.
     * It processes a batch of packets from the queue to improve throughput.
     */
    private void processSendQueue() {
        int processedCount = 0;
        int maxBatchSize = 100; // Process up to 100 packets per interval

        while (processedCount < maxBatchSize) {
            RetryablePacket job = sendQueue.poll(); // Non-blocking retrieval
            if (job == null) {
                break; // Queue is empty, stop for this interval
            }

            processedCount++;
            processSinglePacket(job);
        }

        if (processedCount > 0) {
            logger.debug("[TCP_Service] 📦 Processed {} packets from send queue", processedCount);
        }
    }

    /**
     * Processes a single packet job from the queue.
     * Handles send success, failure, and retry logic.
     */
    private void processSinglePacket(RetryablePacket job) {
        // Attempt to send over the socket
        boolean success = attemptSendInternal(job);

        if (success) {
            // ====================================================
            // === SEND SUCCESSFUL ===
            // Account for TRANSMIT (TX) cost and create HopRecord.
            // ====================================================
            logger.debug("[TCP_Service] ✅ Packet {} sent successfully → {} | Accounting TX...",
                    job.packet().getPacketId(), job.destinationDesc());

            // Call NodeService to account for TX cost -> returns TX delay
            double txDelay = nodeService.processSuccessfulSend(job.originalNodeId(), job.packet());

            // If context exists, create the HopRecord with actual, simulated delays
            if (job.hopContext() != null && !job.packet().isDropped()) {
                HopContext ctx = job.hopContext();
                // Total hop delay = (RX/CPU) + (TX) + (Propagation, etc.)
                double totalHopDelay = ctx.rxCpuDelay() + txDelay;

                PacketHelper.createHopRecordWithActualDelay(
                        job.packet(),
                        ctx.currentNode(),
                        ctx.nextNode(),
                        totalHopDelay, // ✅ Actual simulated delay for the entire hop
                        ctx.routeInfo());

                logger.debug("[TCP_Service] 📝 HopRecord created for Packet {} | Total Hop Delay: {}ms (RX/CPU: {} + TX: {})",
                        job.packet().getPacketId(),
                        String.format("%.2f", totalHopDelay),
                        String.format("%.2f", ctx.rxCpuDelay()),
                        String.format("%.2f", txDelay));
            }

            // ✅ IF SENT TO USER, calculate analysis data and save to DB
            if (job.destinationDesc().startsWith("USER:")) {
                // Calculate final analytics
                PacketHelper.calculateAnalysisData(job.packet());
                logger.info(
                        "[TCP_Service] 📊 AnalysisData calculated for Packet {} | Total Hops: {} | Total Distance: {} km | Total Latency: {} ms",
                        job.packet().getPacketId(),
                        job.packet().getHopRecords() != null ? job.packet().getHopRecords().size() : 0,
                        job.packet().getAnalysisData() != null
                                ? String.format("%.2f", job.packet().getAnalysisData().getTotalDistanceKm())
                                : "N/A",
                        job.packet().getAnalysisData() != null
                                ? String.format("%.2f", job.packet().getAnalysisData().getTotalLatencyMs())
                                : "N/A");

                // ✅ SAVE to TwoPacket + BatchPacket collections
                try {
                    batchPacketService.savePacket(job.packet());
                    logger.info("[TCP_Service] 💾 Saved packet {} to TwoPacket + BatchPacket collections",
                            job.packet().getPacketId());
                } catch (Exception e) {
                    logger.error("[TCP_Service] ❌ Failed to save packet to BatchPacket: {}", e.getMessage(), e);
                }
            }

            // ✅ LOG successful packet to file
            logSuccessfulPacket(job.packet(), job.destinationDesc());

        } else {
            // === SEND FAILED (I/O Error) ===
            // ⚠️ IMPORTANT: Update packet state even on failure
            // In the simulation, the packet still "cost" time and resources.
            handleFailedSend(job);

            if (job.attemptCount() < MAX_RETRIES) {
                // Still have retries left
                logger.warn("[TCP_Service] Send failed for packet {} (attempt {}). Retrying... | TTL: {} | Delay: {}ms",
                        job.packet().getPacketId(), job.attemptCount(),
                        job.packet().getTTL(),
                        String.format("%.2f", job.packet().getAccumulatedDelayMs()));

                // Check if TTL expired after the failed attempt
                if (job.packet().getTTL() <= 0) {
                    logger.error("[TCP_Service] ❌ DROP packet {} due to TTL=0 after send failure.",
                            job.packet().getPacketId());
                    job.packet().setDropped(true);
                    job.packet().setDropReason("TTL_EXPIRED_AFTER_SEND_FAILURE");
                    return; // Do not re-enqueue
                }

                // Create a new job with an incremented attempt count
                RetryablePacket nextAttempt = new RetryablePacket(
                        job.originalNodeId(),
                        job.packet(),
                        job.host(),
                        job.port(),
                        job.destinationDesc(),
                        job.attemptCount() + 1,
                        job.hopContext() // ✅ Pass the original context along
                );
                sendQueue.add(nextAttempt); // Add back to the queue

            } else {
                // Out of retries
                logger.error(
                        "[TCP_Service] ❌ DROP packet {} to {}: Exceeded max {} retries. | TTL: {} | Delay: {}ms",
                        job.packet().getPacketId(), job.destinationDesc(), MAX_RETRIES,
                        job.packet().getTTL(),
                        String.format("%.2f", job.packet().getAccumulatedDelayMs()));

                // Mark packet as dropped
                job.packet().setDropped(true);
                job.packet().setDropReason("TCP_SEND_FAILED_MAX_RETRIES");

                // ✅ SAVE DROPPED PACKET to the database for analysis
                if (job.destinationDesc().startsWith("USER:")) {
                    try {
                        // Calculate AnalysisData before saving
                        if (job.packet().getAnalysisData() == null) {
                            PacketHelper.calculateAnalysisData(job.packet());
                        }

                        // ✅ Save to TwoPacket + BatchPacket
                        batchPacketService.savePacket(job.packet());
                        logger.info("[TCP_Service] 💾 Saved DROPPED packet {} to BatchPacket collections",
                                job.packet().getPacketId());
                    } catch (Exception e) {
                        logger.error("[TCP_Service] ❌ Failed to save dropped packet to database: {}",
                                e.getMessage(), e);
                    }
                }
            }
        }
    } // End processSinglePacket()

    /**
     * Handles the simulation logic for a failed send attempt.
     * In the simulation, a failed attempt still consumes time (latency) and a
     * TTL hop.
     *
     * ✅ Latency is calculated based on the simulation model:
     * - Node -> Node: Based on distance, bandwidth, packet size (from HopContext)
     * - Node -> User: Fixed constant (no HopContext)
     */
    private void handleFailedSend(RetryablePacket job) {
        Packet packet = job.packet();

        // 1. Decrement TTL (the hop was attempted)
        int currentTTL = packet.getTTL();
        packet.setTTL(currentTTL - 1);

        // 2. Increment latency based on simulation model
        double failedAttemptLatency;

        if (job.hopContext() != null) {
            // === CASE 1: Node -> Node (has HopContext) ===
            // Calculate delay using the formula: RX/CPU + TX + Propagation
            HopContext ctx = job.hopContext();

            // Calculate transmission + propagation delay
            NodeInfo currentNode = ctx.currentNode();
            NodeInfo nextNode = ctx.nextNode();

            double bandwidthMHz = currentNode.getCommunication().getBandwidthMHz();
            double bandwidthBps = bandwidthMHz * SimulationConstants.MBPS_TO_BPS_CONVERSION;
            double bandwidthBpms = bandwidthBps / 1000.0; // Bytes per millisecond

            double transmissionDelayMs = (bandwidthBpms > 0)
                    ? packet.getPayloadSizeByte() / bandwidthBpms
                    : 0.0;

            // Calculate distance and propagation delay
            double distanceKm = calculateDistance(currentNode, nextNode);
            double propagationDelayMs = distanceKm / SimulationConstants.PROPAGATION_DIVISOR_KM_MS;

            // Weather impact
            double weatherImpact = 1.0;
            if (currentNode.getWeather() != null) {
                weatherImpact = 1.0 + currentNode.getWeather().getTypicalAttenuationDb()
                        / SimulationConstants.WEATHER_DB_TO_FACTOR;
            }

            // Total delay = RX/CPU (pre-calculated) + TX + Propagation
            failedAttemptLatency = ctx.rxCpuDelay()
                    + (transmissionDelayMs * weatherImpact)
                    + propagationDelayMs;

            logger.debug("[TCP_Service] 🔄 Failed send Node→Node: Tx={}ms, Prop={}ms, Total={}ms",
                    String.format("%.2f", transmissionDelayMs * weatherImpact),
                    String.format("%.2f", propagationDelayMs),
                    String.format("%.2f", failedAttemptLatency));

        } else {
            // === CASE 2: Node -> User (no HopContext) ===
            // Use a fixed constant
            failedAttemptLatency = SimulationConstants.NODE_TO_USER_DELIVERY_DELAY_MS;

            logger.debug("[TCP_Service] 🔄 Failed send Node→User: Delay={}ms (constant)",
                    String.format("%.2f", failedAttemptLatency));
        }

        // Update accumulated delay
        double currentDelay = packet.getAccumulatedDelayMs();
        packet.setAccumulatedDelayMs(currentDelay + failedAttemptLatency);

        logger.debug("[TCP_Service] 📊 Updated packet {} after failed send: TTL {} → {} | Delay {} → {}ms",
                packet.getPacketId(), currentTTL, packet.getTTL(),
                String.format("%.2f", currentDelay),
                String.format("%.2f", packet.getAccumulatedDelayMs()));
    }

    /**
     * Calculates the distance between two nodes (in km) using the Haversine
     * formula.
     */
    private double calculateDistance(NodeInfo from, NodeInfo to) {
        double R = 6371; // Earth radius in km
        double dLat = Math.toRadians(to.getPosition().getLatitude() - from.getPosition().getLatitude());
        double dLon = Math.toRadians(to.getPosition().getLongitude() - from.getPosition().getLongitude());
        double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(Math.toRadians(from.getPosition().getLatitude())) *
                        Math.cos(Math.toRadians(to.getPosition().getLatitude())) *
                        Math.sin(dLon / 2) * Math.sin(dLon / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }

    /**
     * Logs detailed packet information upon successful delivery.
     */
    private void logSuccessfulPacket(Packet packet, String destination) {
        logger.info("═══════════════════════════════════════════════════════════════");
        logger.info("✅ PACKET GỬI THÀNH CÔNG"); // Log string kept in Vietnamese
        logger.info("───────────────────────────────────────────────────────────────");
        logger.info("📦 Packet ID:           {}", packet.getPacketId());
        logger.info("📍 Đích:                {}", destination);
        logger.info("🔄 Node hiện tại:       {}", packet.getCurrentHoldingNodeId());
        logger.info("🎯 Trạm đích:           {}", packet.getStationDest());
        logger.info("👤 User đích:           {}", packet.getDestinationUserId());
        logger.info("⏱️  TTL còn lại:         {}", packet.getTTL());
        logger.info("📈 Delay tích lũy:      {} ms", String.format("%.2f", packet.getAccumulatedDelayMs()));
        logger.info("📊 Max latency cho phép: {} ms", String.format("%.2f", packet.getMaxAcceptableLatencyMs()));
        logger.info("🛣️  Đường đi:            {}", packet.getPathHistory() != null ?
                String.join(" → ", packet.getPathHistory()) : "N/A");
        logger.info("🔧 Service QoS:         {}", packet.getServiceQoS());
        logger.info("🤖 Sử dụng RL:          {}", packet.isUseRL() ? "✓" : "✗");
        if (packet.getHopRecords() != null && !packet.getHopRecords().isEmpty()) {
            logger.info("📝 Số hop đã đi:        {}", packet.getHopRecords().size());
        }
        logger.info(packet.toString());
        logger.info("═══════════════════════════════════════════════════════════════");
    }

    /**
     * Converts an integer to a 4-byte array in big-endian format.
     * This is used for the length-prefixing protocol.
     *
     * @param value The integer value to convert
     * @return A 4-byte array representing the integer
     */
    private byte[] intToBytes(int value) {
        return new byte[] {
            (byte) (value >> 24),
            (byte) (value >> 16),
            (byte) (value >> 8),
            (byte) value
        };
    }

    /**
     * (Consumer I/O)
     * The actual I/O function: Attempts to serialize and send the packet via a
     * Socket.
     *
     * @return true on success, false on a retryable I/O failure.
     */
    private boolean attemptSendInternal(RetryablePacket job) {
        byte[] packetData;
        try {
            // 1. Serialize the packet object to a byte array
            packetData = objectMapper.writeValueAsBytes(job.packet());
        } catch (IOException e) {
            // ✅ OPTIMIZATION 2: Handle non-retryable serialization failure
            // FATAL: This is a non-retryable error. The packet is malformed.
            logger.error(
                    "[TCP_Service] FATAL: Failed to serialize packet {}. Dropping permanently. Error: {}",
                    job.packet().getPacketId(), e.getMessage(), e);

            // Mark as dropped
            job.packet().setDropped(true);
            job.packet().setDropReason("PACKET_SERIALIZATION_FAILED");

            // If this was heading to a user, save it for analysis
            if (job.destinationDesc().startsWith("USER:")) {
                try {
                    PacketHelper.calculateAnalysisData(job.packet()); // Calculate what we can
                    batchPacketService.savePacket(job.packet());
                    logger.info("[TCP_Service] 💾 Saved DROPPED (unserializable) packet {} to database.",
                            job.packet().getPacketId());
                } catch (Exception dbException) {
                    logger.error("[TCP_Service] ❌ Failed to save unserializable/dropped packet {}: {}",
                            job.packet().getPacketId(), dbException.getMessage(), dbException);
                }
            }
            return true; // Return 'true' to signal the job is "complete" and should not be retried.
        }

        logger.debug("[TCP_Service] Sending (Attempt {}/{}): Packet {} to {} at {}:{}...",
                job.attemptCount(), MAX_RETRIES,
                job.packet().getPacketId(), job.destinationDesc(), job.host(), job.port());

        // 2. Open Socket and Send
        // (Use try-with-resources to ensure socket/streams are always closed)
        try (Socket socket = new Socket()) {
            // Add a connect timeout to prevent hanging on an unresponsive host
            socket.connect(
                    new InetSocketAddress(job.host(), job.port()),
                    SimulationConstants.TCP_CONNECT_TIMEOUT_MS // e.g., 1000ms
            );

            try (OutputStream out = socket.getOutputStream()) {
                // Write the 4-byte length prefix (for the protocol)
                out.write(intToBytes(packetData.length));
                
                // Write the packet data
                out.write(packetData);
                out.flush();
                logger.info("[TCP_Service] Successfully sent Packet {} to {}.",
                        job.packet().getPacketId(), job.destinationDesc());
                return true; // Success!
            }
        } catch (IOException e) {
            // Network error (timeout, connection refused, etc.)
            logger.warn("[TCP_Service] I/O error sending packet {} (attempt {}): {}",
                    job.packet().getPacketId(), job.attemptCount(), e.getMessage());
            return false; // Failed, will be retried
        }
    }

    /**
     * Stops the send scheduler during application shutdown.
     */
    public void stop() {
        logger.info("[TCP_Service] Shutting down Send Scheduler...");
        this.retryScheduler.shutdown();
        try {
            if (!this.retryScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                this.retryScheduler.shutdownNow();
            }
            logger.info("[TCP_Service] Send Scheduler stopped.");
        } catch (InterruptedException e) {
            this.retryScheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Selects the best route using RL or a standard fallback.
     */
    private RouteInfo getBestRoute(Packet packet) {
        if (!packet.isUseRL()) {
            return routingService.getBestRoute(packet.getCurrentHoldingNodeId(), packet.getStationDest());
        } else {
            // Attempt to get a route from the RL service
            RouteInfo routeInfo = rlRoutingService.getNextHop(
                    new RoutingRequest(
                            packet.getPacketId(),
                            packet.getCurrentHoldingNodeId(),
                            packet.getStationDest(),
                            packet.getMaxAcceptableLatencyMs(),
                            packet.getTTL(),
                            packet.getServiceQoS()));
            
            if (routeInfo != null) {
                return routeInfo; // Use RL route
            }
            
            // Fallback to standard routing if RL service fails or returns null
            logger.warn("[TCP_Service] RL service returned no route for {}. Falling back to standard routing.", packet.getPacketId());
            return routingService.getBestRoute(packet.getCurrentHoldingNodeId(), packet.getStationDest());
        }
    }
}