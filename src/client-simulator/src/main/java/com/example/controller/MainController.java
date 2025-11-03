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
 * Controller wiring the MainView and model classes.
 * - Handles Send button by creating a Packet from form fields and calling PacketSender
 * - Handles Listen button to start/stop PacketReceiver and update the received list in real-time
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
            Packet p = buildPacketFromForm();
            
            // Send to stationSource node (not directly to destination user)
            String stationSourceId = p.getStationSource();
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
            
            // Send packet to station source node
            sender.send(host, port, p);
            view.lblStatus.setText("Sent packet " + p.getPacketId() + " to station " + stationSourceId + " (" + host + ":" + port + ")");
            
        } catch (Exception ex) {
            view.lblStatus.setText("Send failed: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    private void onListenToggle() {
        if (!listening) {
            try {
                int port = Integer.parseInt(view.tfListenPort.getText().trim());
                receiver.start(port, this::onPacketReceived);
                listening = true;
                view.btnListen.setText("Stop");
                view.lblStatus.setText("Listening on port " + port);
            } catch (IOException ex) {
                view.lblStatus.setText("Failed to listen: " + ex.getMessage());
                ex.printStackTrace();
            }
        } else {
            receiver.stop();
            listening = false;
            view.btnListen.setText("Listen");
            view.lblStatus.setText("Stopped listening");
        }
    }

    /**
     * Load all usernames from the database and populate the ComboBoxes.
     * This method queries all users and extracts their userId (username) to populate the dropdown.
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
     * Called when sender username is selected. Fetches user info from DB,
     * fills sourceUserId, sets listen port (where sender receives packets), and computes stationSource (nearest node).
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
                
                // Set listen port based on sender's port (where this user listens for incoming packets)
                view.tfListenPort.setText(String.valueOf(u.getPort()));

                // Compute nearest node (stationSource) based on user's city
                NodeInfo nearest = nodeService.getNearestNode(u.getUserId());
                if (nearest != null) {
                    view.tfStationSource.setText(nearest.getNodeId());
                    view.lblStatus.setText("Sender: " + u.getUserId() + " (listen on port " + u.getPort() + ") -> Station: " + nearest.getNodeId());
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
     * Called when destination username is selected. Fetches user info from DB,
     * fills destinationUserId, sets send host/port (where to send the packet), and computes stationDest (nearest node).
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
                
                // Set send host/port to destination user's IP and port (where to send the packet)
                if (u.getIpAddress() != null) {
                    view.tfSendHost.setText(u.getIpAddress());
                }
                view.tfSendPort.setText(String.valueOf(u.getPort()));

                // Compute nearest node (stationDest) based on user's city
                NodeInfo nearest = nodeService.getNearestNode(u.getUserId());
                if (nearest != null) {
                    view.tfStationDest.setText(nearest.getNodeId());
                    view.lblStatus.setText("Destination: " + u.getUserId() + " (send to " + u.getIpAddress() + ":" + u.getPort() + ") -> Station: " + nearest.getNodeId());
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
        // Update UI on JavaFX thread
        Platform.runLater(() -> {
            receivedItems.add(0, p); // newest first
            view.lblStatus.setText("Received " + p.getPacketId());
        });
    }

    private Packet buildPacketFromForm() {
        Packet p = new Packet();
        
        // Auto-generate UUID for packetId
        p.setPacketId(java.util.UUID.randomUUID().toString());
        
        // Auto-filled from username selection
        p.setSourceUserId(view.tfSourceUserId.getText());
        p.setDestinationUserId(view.tfDestinationUserId.getText());
        p.setStationSource(view.tfStationSource.getText());
        p.setStationDest(view.tfStationDest.getText());
        
        // Set timestamp to current time
        p.setTimeSentFromSourceMs(System.currentTimeMillis());
        
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

        // Optional TTL (default to 64 if not specified)
        try { 
            String ttlStr = view.tfTTL.getText().trim();
            if (!ttlStr.isEmpty()) {
                p.setTTL(Integer.parseInt(ttlStr));
            } else {
                p.setTTL(64); // default TTL
            }
        } catch (Exception ignored) {
            p.setTTL(64);
        }
        
        // Optional priority override
        try { 
            String prioStr = view.tfPriorityLevel.getText().trim();
            if (!prioStr.isEmpty()) {
                p.setPriorityLevel(Integer.parseInt(prioStr));
            }
        } catch (Exception ignored) {}
        
        // Use RL flag
        p.setUseRL(view.cbUseRL.isSelected());
        
        // dropReason is null by default (packet not dropped when created)
        p.setDropReason(null);
        p.setDropped(false);
        
        // currentHoldingNodeId and nextHopNodeId are auto-managed by network layer
        p.setCurrentHoldingNodeId(null);
        p.setNextHopNodeId(null);

        return p;
    }
}
