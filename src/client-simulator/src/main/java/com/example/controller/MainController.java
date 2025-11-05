package com.example.controller;

import com.example.model.Packet;
import com.example.view.MainView;
import com.example.factory.QoSProfileFactory;
import com.example.model.ServiceQoS;
import com.example.model.ServiceType;
import com.example.repository.IUserRepository;
import com.example.repository.INodeRepository;
import com.example.repository.MongoUserRepository;
import com.example.repository.MongoNodeRepository;
import com.example.model.UserInfo;
import com.example.model.NodeInfo;
import com.example.service.NodeService;
import com.example.service.PacketReceiver;
import com.example.service.PacketSender;

import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;

import java.io.IOException;
import java.util.Base64;

/**
 * Controller with dual packet sending (RL + non-RL) for comparison
 */
public class MainController {

    private final MainView view;
    private final PacketSender sender = new PacketSender();
    private final PacketReceiver receiver = new PacketReceiver();
    private final ObservableList<Packet> receivedItems = FXCollections.observableArrayList();

    private volatile boolean listening = false;

    public MainController(MainView view) {
        this.view = view;
        this.view.lvReceived.setItems(receivedItems);
        // initialize repositories (Mongo-backed). In production, inject these.
        try {
            this.userRepo = new MongoUserRepository();
        } catch (Exception ex) {
            this.userRepo = null;
            System.err.println("Failed to initialize MongoUserRepository: " + ex.getMessage());
        }
        try {
            this.nodeRepo = new MongoNodeRepository();
        } catch (Exception ex) {
            this.nodeRepo = null;
            System.err.println("Failed to initialize MongoNodeRepository: " + ex.getMessage());
        }
        if (this.userRepo != null && this.nodeRepo != null) {
            this.nodeService = new NodeService(this.userRepo, this.nodeRepo);
        }
        attachHandlers();
        loadUsernames();
    }

    private void attachHandlers() {
        view.btnSend.setOnAction(e -> onSend());
        view.btnListen.setOnAction(e -> onListenToggle());
        view.btnClearLog.setOnAction(e -> onClearLog());
        
        // When sender username is selected, fetch user info and compute stationSource
        view.cbSenderUsername.valueProperty().addListener((obs, oldV, newV) -> {
            if (newV != null && !newV.trim().isEmpty()) {
                onSenderUsernameSelected(newV.trim());
            }
        });
        
        // When destination username is selected, fetch user info and compute stationDest
        view.cbDestinationUsername.valueProperty().addListener((obs, oldV, newV) -> {
            if (newV != null && !newV.trim().isEmpty()) {
                onDestinationUsernameSelected(newV.trim());
            }
        });
        
        // When service type is selected, display detailed QoS information
        view.cbServiceType.valueProperty().addListener((obs, oldV, newV) -> {
            if (newV != null) {
                onServiceTypeSelected(newV);
            }
        });
    }

    // repositories and nodeService (optional if Mongo not available)
    private IUserRepository userRepo;
    private INodeRepository nodeRepo;
    private NodeService nodeService;

    private void onSend() {
        try {
            // Get number of packets to send
            int packetCount = 1;
            try {
                String countStr = view.tfPacketCount.getText().trim();
                if (!countStr.isEmpty()) {
                    packetCount = Integer.parseInt(countStr);
                    if (packetCount < 1) packetCount = 1;
                    if (packetCount > 1000) packetCount = 1000; // safety limit
                }
            } catch (Exception ex) {
                packetCount = 1;
            }
            
            // Build base packet template
            Packet basePacket = buildPacketFromForm();

            // Validate destination user is set
            if (basePacket.getDestinationUserId() == null || basePacket.getDestinationUserId().trim().isEmpty()) {
                view.lblStatus.setText("Destination user not set. Please select destination username.");
                return;
            }
            
            // Send to stationSource node (not directly to destination user)
            String stationSourceId = basePacket.getStationSource();
            if (stationSourceId == null || stationSourceId.trim().isEmpty()) {
                view.lblStatus.setText("Station source not set. Please select sender username.");
                return;
            }
            
            // Load stationSource node from database to get its communication info
            if (nodeRepo == null) {
                view.lblStatus.setText("Node repository not available");
                return;
            }
            
            var nodeOpt = nodeRepo.getNodeInfo(stationSourceId);
            if (nodeOpt.isEmpty()) {
                view.lblStatus.setText("Station source node not found: " + stationSourceId);
                return;
            }
            
            NodeInfo stationNode = nodeOpt.get();
            if (stationNode.getCommunication() == null) {
                view.lblStatus.setText("Station node has no communication info: " + stationSourceId);
                return;
            }
            
            // Get IP and port from station node's communication
            String host = stationNode.getCommunication().getIpAddress();
            int port = stationNode.getCommunication().getPort();
            
            if (host == null || host.trim().isEmpty()) {
                view.lblStatus.setText("Station node has no IP address: " + stationSourceId);
                return;
            }
            
            final String finalHost = host;
            final int finalPort = port;
            final int finalPacketCount = packetCount;
            final int totalPackets = packetCount * 2; // Each count = 2 packets (1 RL + 1 non-RL)
            
            view.lblStatus.setText("Sending " + finalPacketCount + " pairs (" + totalPackets + " packets total)...");
            
            // Create a thread pool for parallel sending
            java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(
                Math.min(totalPackets, 20) // max 20 concurrent threads
            );
            
            java.util.concurrent.atomic.AtomicInteger successCount = new java.util.concurrent.atomic.AtomicInteger(0);
            java.util.concurrent.atomic.AtomicInteger failCount = new java.util.concurrent.atomic.AtomicInteger(0);
            
            // Send pairs of packets (each pair has same packetId, different useRL)
            for (int i = 0; i < finalPacketCount; i++) {
                // Generate ONE shared packetId for this PAIR
                final String sharedPacketId = java.util.UUID.randomUUID().toString();
                
                // Send RL version
                executor.submit(() -> {
                    try {
                        Packet rlPacket = clonePacket(basePacket);
                        rlPacket.setPacketId(sharedPacketId);
                        rlPacket.setUseRL(true);
                        
                        System.out.println("🤖 Sending RL packet: ID=" + sharedPacketId + ", useRL=" + rlPacket.isUseRL());
                        
                        PacketSender threadSender = new PacketSender();
                        threadSender.send(finalHost, finalPort, rlPacket);
                        successCount.incrementAndGet();
                    } catch (Exception ex) {
                        failCount.incrementAndGet();
                        ex.printStackTrace();
                    }
                });
                
                // Send non-RL version (same packetId, different useRL)
                executor.submit(() -> {
                    try {
                        Packet nonRlPacket = clonePacket(basePacket);
                        nonRlPacket.setPacketId(sharedPacketId);
                        nonRlPacket.setUseRL(false);
                        
                        System.out.println("📍 Sending non-RL packet: ID=" + sharedPacketId + ", useRL=" + nonRlPacket.isUseRL());
                        
                        PacketSender threadSender = new PacketSender();
                        threadSender.send(finalHost, finalPort, nonRlPacket);
                        successCount.incrementAndGet();
                    } catch (Exception ex) {
                        failCount.incrementAndGet();
                        ex.printStackTrace();
                    }
                });
            }
            
            // Shutdown executor and wait for completion in background
            executor.shutdown();
            new Thread(() -> {
                try {
                    executor.awaitTermination(60, java.util.concurrent.TimeUnit.SECONDS);
                    Platform.runLater(() -> {
                        view.lblStatus.setText(String.format(
                            "Sent %d pairs (%d packets) to station %s - Success: %d, Failed: %d",
                            finalPacketCount, totalPackets, stationSourceId, 
                            successCount.get(), failCount.get()
                        ));
                    });
                } catch (InterruptedException ex) {
                    Platform.runLater(() -> view.lblStatus.setText("Packet sending interrupted"));
                }
            }).start();
            
        } catch (Exception ex) {
            view.lblStatus.setText("Send failed: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    /**
     * Clone a packet to create independent copies for RL and non-RL versions
     */
    private Packet clonePacket(Packet original) {
        Packet clone = new Packet();
        
        // Copy all fields (packetId will be set separately)
        clone.setSourceUserId(original.getSourceUserId());
        clone.setDestinationUserId(original.getDestinationUserId());
        clone.setStationSource(original.getStationSource());
        clone.setStationDest(original.getStationDest());
        clone.setTimeSentFromSourceMs(System.currentTimeMillis()); // Fresh timestamp
        clone.setPayloadDataBase64(original.getPayloadDataBase64());
        clone.setPayloadSizeByte(original.getPayloadSizeByte());
        clone.setServiceQoS(original.getServiceQoS());
        clone.setTTL(original.getTTL());
        clone.setPriorityLevel(original.getPriorityLevel());
        clone.setMaxAcceptableLatencyMs(original.getMaxAcceptableLatencyMs());
        clone.setMaxAcceptableLossRate(original.getMaxAcceptableLossRate());
        clone.setDropReason(null);
        clone.setDropped(false);
        clone.setCurrentHoldingNodeId(null);
        clone.setNextHopNodeId(null);
        
        return clone;
    }

    private void onListenToggle() {
        if (!listening) {
            try {
                int port = Integer.parseInt(view.tfListenPort.getText().trim());
                receiver.start(port, this::onPacketReceived);
                listening = true;
                view.btnListen.setText("Stop Listening");
                view.lblStatus.setText("Listening on port " + port);
            } catch (IOException ex) {
                view.lblStatus.setText("Failed to listen: " + ex.getMessage());
                ex.printStackTrace();
            }
        } else {
            receiver.stop();
            listening = false;
            view.btnListen.setText("Start Listening");
            view.lblStatus.setText("Stopped listening");
        }
    }

    private void onClearLog() {
        receivedItems.clear();
        view.lblStatus.setText("Log cleared (" + java.time.LocalTime.now() + ")");
    }

    /**
     * Load all usernames from the database and populate the ComboBoxes.
     */
    private void loadUsernames() {
        if (userRepo == null || nodeRepo == null) {
            view.lblStatus.setText("MongoDB not available - username list not loaded");
            return;
        }
        try {
            var allUsers = userRepo.findAll();
            for (UserInfo u : allUsers) {
                view.cbSenderUsername.getItems().add(u.getUserId());
                view.cbDestinationUsername.getItems().add(u.getUserId());
            }
            view.lblStatus.setText("Loaded " + allUsers.size() + " usernames from DB");
            view.cbSenderUsername.getSelectionModel().selectFirst();
            view.cbDestinationUsername.getSelectionModel().selectFirst();
            view.lblStatus.setText("Ready");

        } catch (Exception ex) {
            view.lblStatus.setText("Error loading usernames: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    /**
     * Called when service type is selected. Displays detailed QoS information.
     */
    private void onServiceTypeSelected(ServiceType serviceType) {
        try {
            ServiceQoS qos = QoSProfileFactory.getQosProfile(serviceType);
            String qosInfo = String.format(
                "Service: %s\n" +
                "Priority: %d\n" +
                "Max Latency: %.1f ms\n" +
                "Max Jitter: %.1f ms\n" +
                "Min Bandwidth: %.1f Mbps\n" +
                "Max Loss Rate: %.2f%%",
                qos.serviceType(),
                qos.defaultPriority(),
                qos.maxLatencyMs(),
                qos.maxJitterMs(),
                qos.minBandwidthMbps(),
                qos.maxLossRate() * 100
            );
            view.lblQoSDetail.setText(qosInfo);
        } catch (Exception ex) {
            view.lblQoSDetail.setText("Failed to load QoS profile: " + ex.getMessage());
        }
    }

    /**
     * Called when sender username is selected.
     */
    private void onSenderUsernameSelected(String username) {
        if (userRepo == null || nodeService == null) {
            view.lblStatus.setText("User/Node repositories not available");
            return;
        }
        try {
            var userOpt = userRepo.findByUserId(username);
            if (userOpt.isPresent()) {
                UserInfo u = userOpt.get();
                
                // Auto-fill sourceUserId
                view.tfSourceUserId.setText(u.getUserId());
                
                // Set listen port based on sender's port
                view.tfListenPort.setText(String.valueOf(u.getPort()));

                // If currently listening, restart receiver on new port
                if (listening) {
                    try {
                        int newPort = Integer.parseInt(view.tfListenPort.getText().trim());
                        receiver.stop();
                        receiver.start(newPort, this::onPacketReceived);
                        view.lblStatus.setText("Listening restarted on port " + newPort + " for sender " + u.getUserId());
                        view.btnListen.setText("Stop Listening");
                        listening = true;
                    } catch (Exception ex) {
                        // Could not restart listener - report and set listening=false so user can retry
                        listening = false;
                        view.btnListen.setText("Start Listening");
                        view.lblStatus.setText("Failed to restart listener on new port: " + ex.getMessage());
                        ex.printStackTrace();
                    }
                }

                // Compute nearest node (stationSource)
                NodeInfo nearest = nodeService.getNearestNode(u.getUserId());
                if (nearest != null) {
                    view.tfStationSource.setText(nearest.getNodeId());
                    view.lblStatus.setText("Sender: " + u.getUserId() + " (port " + u.getPort() + ") -> Station: " + nearest.getNodeId());
                } else {
                    view.lblStatus.setText("No nearest node found for sender " + username);
                }
            } else {
                view.lblStatus.setText("Sender user not found: " + username);
            }
        } catch (Exception ex) {
            view.lblStatus.setText("Error fetching sender: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    /**
     * Called when destination username is selected.
     */
    private void onDestinationUsernameSelected(String username) {
        if (userRepo == null || nodeService == null) {
            view.lblStatus.setText("User/Node repositories not available");
            return;
        }
        try {
            var userOpt = userRepo.findByUserId(username);
            if (userOpt.isPresent()) {
                UserInfo u = userOpt.get();
                
                // Auto-fill destinationUserId
                view.tfDestinationUserId.setText(u.getUserId());

                // Set send host/port to destination user's IP and port
                if (u.getIpAddress() != null) {
                    view.tfSendHost.setText(u.getIpAddress());
                }
                view.tfSendPort.setText(String.valueOf(u.getPort()));

                // Compute nearest node (stationDest)
                NodeInfo nearest = nodeService.getNearestNode(u.getUserId());
                if (nearest != null) {
                    view.tfStationDest.setText(nearest.getNodeId());
                    view.lblStatus.setText("Destination: " + u.getUserId() + " (" + u.getIpAddress() + ":" + u.getPort() + ") -> Station: " + nearest.getNodeId());
                } else {
                    view.lblStatus.setText("No nearest node found for destination " + username);
                }
            } else {
                view.lblStatus.setText("Destination user not found: " + username);
            }
        } catch (Exception ex) {
            view.lblStatus.setText("Error fetching destination: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    private void onPacketReceived(Packet p) {
        // Debug log
        System.out.println("📥 CLIENT RECEIVED: ID=" + p.getPacketId() + ", useRL=" + p.isUseRL() + 
            " (from " + p.getSourceUserId() + " to " + p.getDestinationUserId() + ")");
        
        // Update UI on JavaFX thread
        Platform.runLater(() -> {
            receivedItems.add(0, p); // newest first
            view.lblStatus.setText("Received " + p.getPacketId() + (p.isUseRL() ? " (RL)" : " (non-RL)"));
        });
    }

    private Packet buildPacketFromForm() {
        Packet p = new Packet();
        
        // PacketId will be set during sending (same for each pair)
        
        // Auto-filled from username selection
        p.setSourceUserId(view.tfSourceUserId.getText());
        p.setDestinationUserId(view.tfDestinationUserId.getText());
        p.setStationSource(view.tfStationSource.getText());
        p.setStationDest(view.tfStationDest.getText());
        
        // Timestamp will be set during cloning
        
        // Encode payload to base64
        String payload = view.taPayload.getText();
        if (payload != null && !payload.trim().isEmpty()) {
            byte[] bytes = payload.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            String base64 = Base64.getEncoder().encodeToString(bytes);
            p.setPayloadDataBase64(base64);
            
            // Use user-provided packet size if specified, otherwise use actual payload size
            try {
                String sizeStr = view.tfPayloadSizeByte.getText().trim();
                if (!sizeStr.isEmpty()) {
                    p.setPayloadSizeByte(Integer.parseInt(sizeStr));
                } else {
                    p.setPayloadSizeByte(bytes.length);
                }
            } catch (Exception ex) {
                p.setPayloadSizeByte(bytes.length);
            }
        } else {
            // If no payload but size is specified, use the specified size
            try {
                String sizeStr = view.tfPayloadSizeByte.getText().trim();
                if (!sizeStr.isEmpty()) {
                    p.setPayloadSizeByte(Integer.parseInt(sizeStr));
                }
            } catch (Exception ignored) {}
        }

        // Use selected ServiceType to fill ServiceQoS via QoSProfileFactory
        ServiceType selected = view.cbServiceType.getValue();
        if (selected != null) {
            try {
                ServiceQoS qosProfile = QoSProfileFactory.getQosProfile(selected);
                p.setServiceQoS(qosProfile);
                // Set priority from QoS profile default
                p.setPriorityLevel(qosProfile.defaultPriority());
                // Set QoS constraints
                p.setMaxAcceptableLatencyMs(qosProfile.maxLatencyMs());
                p.setMaxAcceptableLossRate(qosProfile.maxLossRate());
            } catch (Exception ex) {
                // ignore and leave serviceQoS null
            }
        }

        // TTL default is 30
        p.setTTL(30);
        
        // dropReason is null by default
        p.setDropReason(null);
        p.setDropped(false);
        
        // currentHoldingNodeId and nextHopNodeId are auto-managed by network layer
        p.setCurrentHoldingNodeId(null);
        p.setNextHopNodeId(null);

        return p;
    }
}