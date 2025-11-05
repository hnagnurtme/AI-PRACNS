package com.sagin.network.implement;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sagin.model.NodeInfo;
import com.sagin.model.Packet;
import com.sagin.network.interfaces.INodeGateway;
import com.sagin.network.interfaces.ITCP_Service;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Implements the INodeGateway interface.
 * This class opens a TCP server socket to listen for incoming Packets destined for this node.
 * It uses a fixed thread pool to handle concurrent client connections and a
 * length-prefixing protocol for message framing.
 */
public class NodeGateway implements INodeGateway {

    private static final Logger logger = LoggerFactory.getLogger(NodeGateway.class);

    // ObjectMapper is thread-safe, so we can create one static instance.
    private static final ObjectMapper objectMapper = createObjectMapper();

    // --- Constants ---
    /**
     * Maximum number of worker threads to handle client connections.
     * This prevents resource exhaustion (e.g., OutOfMemoryError).
     */
    private static final int MAX_WORKER_THREADS = 100;
    
    /**
     * Maximum allowed size for an incoming packet (e.g., 16KB).
     * This prevents OutOfMemoryError from a malicious client sending an invalid length.
     */
    private static final int MAX_PACKET_SIZE = 16 * 1024; // 16 KB

    private final ITCP_Service tcpService;
    private ServerSocket serverSocket;
    private Thread listenerThread;
    private ExecutorService clientHandlerPool;
    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private String nodeId;

    /**
     * Helper method to initialize the static ObjectMapper.
     */
    private static ObjectMapper createObjectMapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.registerModule(new JavaTimeModule());
        return mapper;
    }

    /**
     * Constructs a new NodeGateway.
     *
     * @param tcpService The service responsible for processing received packets.
     */
    public NodeGateway(ITCP_Service tcpService) {
        this.tcpService = tcpService;
    }

    /**
     * Starts the TCP listener on a specific port.
     * Initializes the ServerSocket and worker threads.
     *
     * @param info The info for the current node (primarily to get the nodeId).
     * @param port The port to listen on.
     */
    @Override
    public void startListening(NodeInfo info, int port) {
        if (info == null || info.getNodeId() == null) {
            logger.error("[NodeGateway] Cannot start: NodeInfo or NodeId is null.");
            return;
        }
        this.nodeId = info.getNodeId();

        // Atomically set isRunning to true only if it's currently false
        if (isRunning.compareAndSet(false, true)) {
            try {
                serverSocket = new ServerSocket(port);
                
                // Use a bounded thread pool to prevent resource exhaustion
                clientHandlerPool = Executors.newFixedThreadPool(MAX_WORKER_THREADS);
                logger.info("[NodeGateway] Node {} listening on port {}...", nodeId, port);

                // Create and start the main listener thread
                listenerThread = new Thread(this::runListenerLoop, "NodeGateway-Listener-" + nodeId);
                listenerThread.start();

            } catch (IOException e) {
                logger.error("[NodeGateway] Node {}: Failed to open port {}: {}", nodeId, port, e.getMessage());
                isRunning.set(false); // Rollback state if startup failed
            }
        } else {
            logger.warn("[NodeGateway] Node {} is already running on port {}.", nodeId, serverSocket.getLocalPort());
        }
    }

    /**
     * The main loop for the listener thread.
     * Continuously accepts new connections and submits them to the worker pool.
     */
    private void runListenerLoop() {
        while (isRunning.get()) {
            try {
                // Blocks until a new connection is made
                Socket clientSocket = serverSocket.accept();
                logger.debug("[NodeGateway] Node {}: Accepted connection from {}", nodeId, clientSocket.getRemoteSocketAddress());
                
                // Submit the client handling task to the worker pool
                clientHandlerPool.submit(() -> handleClient(clientSocket));
                
            } catch (SocketException se) {
                // This is expected when stopListening() closes the serverSocket
                if (isRunning.get()) { // Log only if not an intentional stop
                    logger.error("[NodeGateway] Node {}: SocketException on accept: {}", nodeId, se.getMessage());
                } else {
                    logger.info("[NodeGateway] Node {}: Listener loop stopping.", nodeId);
                }
            } catch (IOException e) {
                if (isRunning.get()){
                    logger.error("[NodeGateway] Node {}: I/O error on accept: {}", nodeId, e.getMessage());
                }
            }
        }
        logger.info("[NodeGateway] Node {}: Listener thread terminated.", nodeId);
    }

    /**
     * Handles the entire lifecycle of a single client connection.
     * This method reads packets using a length-prefixing protocol.
     * It can read multiple packets from a single, persistent connection.
     *
     * Protocol: [4-byte integer length N][N bytes of JSON data]
     *
     * @param clientSocket The connected client socket.
     */
    public void handleClient(Socket clientSocket) {
        // Use try-with-resources to automatically close the socket and streams
        try (clientSocket; DataInputStream dis = new DataInputStream(clientSocket.getInputStream())) {

            // Keep reading packets from this connection as long as it's open
            while (isRunning.get()) {
                int packetLength;
                try {
                    // 1. Read the 4-byte integer length prefix
                    packetLength = dis.readInt();
                } catch (EOFException eof) {
                    // Clean shutdown: client closed the connection.
                    logger.debug("[NodeGateway] Node {}: Client {} closed the connection.", nodeId, clientSocket.getRemoteSocketAddress());
                    break; // Exit the while loop
                }

                // 2. Sanity check the length
                if (packetLength <= 0 || packetLength > MAX_PACKET_SIZE) {
                    logger.warn("[NodeGateway] Node {}: Received invalid packet length {} from {}. Closing connection.", 
                                nodeId, packetLength, clientSocket.getRemoteSocketAddress());
                    break; // Invalid length, stop processing this client
                }

                // 3. Read exactly 'packetLength' bytes for the payload
                byte[] data = new byte[packetLength];
                dis.readFully(data); // This blocks until all 'packetLength' bytes are read

                // 4. Deserialize and process the packet
                logger.debug("[NodeGateway] Node {}: Received packet ({} bytes) from {}. Deserializing...", 
                                nodeId, data.length, clientSocket.getRemoteSocketAddress());

                Packet receivedPacket = objectMapper.readValue(data, Packet.class);

                // IMPORTANT: Set this node as the current holder
                receivedPacket.setCurrentHoldingNodeId(this.nodeId);

                // Pass the packet to the service layer
                tcpService.receivePacket(receivedPacket);
            }

        } catch (JsonProcessingException jpe) {
            logger.error("[NodeGateway] Node {}: Failed to deserialize JSON from {}: {}", 
                            nodeId, clientSocket.getRemoteSocketAddress(), jpe.getMessage());
        } catch (IOException e) {
            if (isRunning.get()) { // Don't flood logs during a shutdown
                logger.error("[NodeGateway] Node {}: I/O error handling client {}: {}", 
                            nodeId, clientSocket.getRemoteSocketAddress(), e.getMessage());
            }
        }
        // The socket is automatically closed here by the try-with-resources block.
    }


    /**
     * Stops the listener and performs a graceful shutdown.
     * Closes the ServerSocket and terminates the worker pool.
     */
    @Override
    public void stopListening() {
        if (isRunning.compareAndSet(true, false)) {
            logger.info("[NodeGateway] Node {}: Shutting down...", nodeId);
            
            // Close the server socket to interrupt the blocking accept() call
            try {
                if (serverSocket != null && !serverSocket.isClosed()) {
                    serverSocket.close();
                }
            } catch (IOException e) {
                logger.error("[NodeGateway] Node {}: Error closing ServerSocket: {}", nodeId, e.getMessage());
            }

            // Perform a graceful shutdown of the worker pool
            if (clientHandlerPool != null) {
                clientHandlerPool.shutdown(); // Disable new tasks
                try {
                    // Wait a bit for existing tasks to complete
                    if (!clientHandlerPool.awaitTermination(5, TimeUnit.SECONDS)) {
                        clientHandlerPool.shutdownNow(); // Forcefully stop pending tasks
                    }
                } catch (InterruptedException ie) {
                    clientHandlerPool.shutdownNow();
                    Thread.currentThread().interrupt();
                }
            }

            // Wait for the main listener thread to die
            if (listenerThread != null) {
                try {
                    listenerThread.join(1000); // Wait max 1 second
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            logger.info("[NodeGateway] Node {}: Shutdown complete.", nodeId);
        } else {
            logger.warn("[NodeGateway] Node {}: Gateway was not running.", nodeId);
        }
    }
}