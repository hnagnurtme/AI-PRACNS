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
import javafx.beans.value.ChangeListener; 

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class MainController {

    private final MainView view;
    private final PacketReceiver receiver = new PacketReceiver();
    private final ObservableList<Packet> receivedItems = FXCollections.observableArrayList();

    private volatile boolean listening = false;
    
    // === SỬA LỖI 2: Tạo Thread Pool MỘT LẦN ===
    // Dùng chung thread pool này cho tất cả các tác vụ gửi
    private final ExecutorService sendExecutor = Executors.newFixedThreadPool(20);

    // Repositories and Services
    private final IUserRepository userRepo;
    private final INodeRepository nodeRepo;
    private final NodeService nodeService;
    
    // Listeners (để có thể gỡ ra và gắn vào)
    private final ChangeListener<String> senderListener;
    private final ChangeListener<String> destListener;

    public MainController(MainView view) {
        this.view = view;
        this.view.lvReceived.setItems(receivedItems);

        IUserRepository tempUserRepo = null;
        INodeRepository tempNodeRepo = null;
        
        try {
            tempUserRepo = MongoUserRepository.getInstance(); 
        } catch (Exception ex) {
            System.err.println("Failed to initialize MongoUserRepository: " + ex.getMessage());
        }
        
        try {
            tempNodeRepo = MongoNodeRepository.getInstance();
        } catch (Exception ex) {
            System.err.println("Failed to initialize MongoNodeRepository: " + ex.getMessage());
        }
        
        this.userRepo = tempUserRepo;
        this.nodeRepo = tempNodeRepo;
        
        if (this.userRepo != null && this.nodeRepo != null) {
            this.nodeService = new NodeService(this.userRepo, this.nodeRepo);
        } else {
            this.nodeService = null;
            view.lblStatus.setText("FATAL: Could not load repositories. Check DB connection.");
        }
        
        // Khởi tạo listeners
        this.senderListener = (obs, oldV, newV) -> onSenderUsernameSelected(newV);
        this.destListener = (obs, oldV, newV) -> onDestinationUsernameSelected(newV);

        attachHandlers();
        loadUsernames();
    }

    private void attachHandlers() {
        view.btnSend.setOnAction(e -> onSend());
        view.btnListen.setOnAction(e -> onListenToggle());
        view.btnClearLog.setOnAction(e -> onClearLog());
        
        view.cbSenderUsername.valueProperty().addListener(senderListener);
        view.cbDestinationUsername.valueProperty().addListener(destListener);
        
        view.cbServiceType.valueProperty().addListener((obs, oldV, newV) -> {
            if (newV != null) onServiceTypeSelected(newV);
        });
    }

    private void onSend() {
        try {
            int packetCount;
            try {
                packetCount = Integer.parseInt(view.tfPacketCount.getText().trim());
                packetCount = Math.max(1, Math.min(1000, packetCount)); // Giới hạn 1-1000
            } catch (Exception ex) {
                packetCount = 1;
            }
            
            Packet basePacket = buildPacketFromForm();

            if (basePacket.getDestinationUserId() == null || basePacket.getDestinationUserId().isBlank()) {
                view.lblStatus.setText("Lỗi: Vui lòng chọn người nhận");
                return;
            }
            
            String stationSourceId = basePacket.getStationSource();
            if (stationSourceId == null || stationSourceId.isBlank()) {
                view.lblStatus.setText("Lỗi: Vui lòng chọn người gửi (để tìm trạm nguồn)");
                return;
            }
            
            if (nodeRepo == null) {
                view.lblStatus.setText("Lỗi: Node repository không khả dụng");
                return;
            }
            
            var nodeOpt = nodeRepo.getNodeInfo(stationSourceId);
            if (nodeOpt.isEmpty()) {
                view.lblStatus.setText("Lỗi: Không tìm thấy trạm nguồn " + stationSourceId);
                return;
            }
            
            NodeInfo stationNode = nodeOpt.get();
            if (stationNode.getCommunication() == null || 
                stationNode.getCommunication().getIpAddress() == null || 
                stationNode.getCommunication().getIpAddress().isBlank()) {
                view.lblStatus.setText("Lỗi: Trạm nguồn " + stationSourceId + " thiếu thông tin IP/Port");
                return;
            }
            
            String host = stationNode.getCommunication().getIpAddress();
            int port = stationNode.getCommunication().getPort();
            
            // === THÊM: Kiểm tra server có sẵn sàng không ===
            System.out.println("🔍 Đang kiểm tra kết nối đến " + host + ":" + port + "...");
            if (!com.example.util.NetworkUtils.isServiceAvailable(host, port, 2000)) {
                view.lblStatus.setText("❌ Lỗi: Không thể kết nối đến server " + host + ":" + port + ". Server có đang chạy không?");
                System.err.println("❌ Server " + host + ":" + port + " không phản hồi!");
                return;
            }
            System.out.println("✅ Server " + host + ":" + port + " đã sẵn sàng!");
            
            final int totalPackets = packetCount * 2;
            
            view.lblStatus.setText("Đang gửi " + packetCount + " cặp (" + totalPackets + " packets)...");
            
            // === SỬA LỖI 1: Dùng Singleton MỘT LẦN ===
            final PacketSender sender = PacketSender.getInstance();
            
            AtomicInteger successCount = new AtomicInteger(0);
            AtomicInteger failCount = new AtomicInteger(0);
            
            for (int i = 0; i < packetCount; i++) {
                final String sharedPacketId = java.util.UUID.randomUUID().toString();
                
                
                // Gửi RL version
                sendExecutor.submit(() -> {
                    try {
                        Packet rlPacket = clonePacket(basePacket);
                        rlPacket.setPacketId(sharedPacketId);
                        rlPacket.setUseRL(true);
                        rlPacket.setType("DATA"); // Thêm type field
                        
                        System.out.println("📤 Gửi RL packet: " + sharedPacketId + " -> " + host + ":" + port);
                        sender.send(host, port, rlPacket); 
                        successCount.incrementAndGet();
                        System.out.println("✅ Đã gửi RL packet: " + sharedPacketId);
                    } catch (Exception ex) {
                        failCount.incrementAndGet();
                        System.err.println("❌ LỖI gửi RL packet " + sharedPacketId + ": " + ex.getMessage());
                        ex.printStackTrace();
                    }
                });
                
                // Gửi non-RL version
                sendExecutor.submit(() -> {
                    try {
                        Packet nonRlPacket = clonePacket(basePacket);
                        nonRlPacket.setPacketId(sharedPacketId);
                        nonRlPacket.setUseRL(false);
                        nonRlPacket.setType("DATA"); // Thêm type field
                        
                        System.out.println("📤 Gửi non-RL packet: " + sharedPacketId + " -> " + host + ":" + port);
                        sender.send(host, port, nonRlPacket); 
                        successCount.incrementAndGet();
                        System.out.println("✅ Đã gửi non-RL packet: " + sharedPacketId);
                    } catch (Exception ex) {
                        failCount.incrementAndGet();
                        System.err.println("❌ LỖI gửi non-RL packet " + sharedPacketId + ": " + ex.getMessage());
                        ex.printStackTrace();
                    }
                });
            }
            
            // Dùng một thread riêng để chờ kết quả (không block UI)
            final int finalPacketCount = packetCount;
            new Thread(() -> {
                // Đợi 2 giây để các packet có thời gian được gửi đi
                // (Không dùng awaitTermination vì pool là chung, không được shutdown)
                try { Thread.sleep(2000); } catch (InterruptedException ignored) {}
                
                Platform.runLater(() -> {
                    view.lblStatus.setText(String.format(
                        "Đã gửi %d cặp (%d packets) đến %s - Thành công: %d, Thất bại: %d",
                        finalPacketCount, totalPackets, stationSourceId, 
                        successCount.get(), failCount.get()
                    ));
                });
            }).start();
            
        } catch (Exception ex) {
            view.lblStatus.setText("Gửi thất bại: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    private Packet clonePacket(Packet original) {
        Packet clone = new Packet();
        
        clone.setSourceUserId(original.getSourceUserId());
        clone.setDestinationUserId(original.getDestinationUserId());
        clone.setStationSource(original.getStationSource());
        clone.setStationDest(original.getStationDest());
        clone.setTimeSentFromSourceMs(System.currentTimeMillis()); 
        clone.setPayloadDataBase64(original.getPayloadDataBase64());
        clone.setPayloadSizeByte(original.getPayloadSizeByte());
        clone.setServiceQoS(original.getServiceQoS());
        clone.setTTL(original.getTTL());
        clone.setPriorityLevel(original.getPriorityLevel());
        clone.setMaxAcceptableLatencyMs(original.getMaxAcceptableLatencyMs());
        clone.setMaxAcceptableLossRate(original.getMaxAcceptableLossRate());
        
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

    private void loadUsernames() {
        if (userRepo == null || nodeRepo == null) {
            view.lblStatus.setText("DB không khả dụng - không thể tải username");
            return;
        }
        try {
            var allUsers = userRepo.findAll();
            
            view.cbSenderUsername.valueProperty().removeListener(senderListener);
            view.cbDestinationUsername.valueProperty().removeListener(destListener);
            
            view.cbSenderUsername.getItems().clear();
            view.cbDestinationUsername.getItems().clear();
            
            for (UserInfo u : allUsers) {
                view.cbSenderUsername.getItems().add(u.getUserId());
                view.cbDestinationUsername.getItems().add(u.getUserId());
            }

            view.cbSenderUsername.getSelectionModel().selectFirst();
            view.cbDestinationUsername.getSelectionModel().selectFirst();
            
            view.cbSenderUsername.valueProperty().addListener(senderListener);
            view.cbDestinationUsername.valueProperty().addListener(destListener);
            
            if (!view.cbSenderUsername.getItems().isEmpty()) {
                onSenderUsernameSelected(view.cbSenderUsername.getValue());
            }
            if (!view.cbDestinationUsername.getItems().isEmpty()) {
                onDestinationUsernameSelected(view.cbDestinationUsername.getValue());
            }

            view.lblStatus.setText("Ready. Đã tải " + allUsers.size() + " users.");

        } catch (Exception ex) {
            view.lblStatus.setText("Lỗi tải usernames: " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    private void onServiceTypeSelected(ServiceType serviceType) {
        try {
            ServiceQoS qos = QoSProfileFactory.getQosProfile(serviceType);
            String qosInfo = String.format(
                "Service: %s\n" +
                "Priority: %d\n" +
                "Max Latency: %.1f ms\n" +
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

    private void onSenderUsernameSelected(String username) {
        if (username == null || username.isBlank()) return;
        if (userRepo == null || nodeService == null) return;
        
        try {
            var userOpt = userRepo.findByUserId(username);
            if (userOpt.isPresent()) {
                UserInfo u = userOpt.get();
                
                view.tfListenPort.setText(String.valueOf(u.getPort()));

                if (listening) {
                    try {
                        int newPort = u.getPort();
                        receiver.stop();
                        receiver.start(newPort, this::onPacketReceived);
                        view.lblStatus.setText("Đang nghe trên port mới " + newPort + " của " + u.getUserId());
                        view.btnListen.setText("Stop Listening");
                        listening = true;
                    } catch (Exception ex) {
                        listening = false;
                        view.btnListen.setText("Start Listening");
                        view.lblStatus.setText("Lỗi restart listener trên port " + u.getPort());
                    }
                }

                Optional<NodeInfo> nearestOptional = nodeService.getNearestNode(u.getUserId());
                NodeInfo nearest = nearestOptional.orElse(null);

                if (nearest != null) {
                    view.lblStatus.setText("Người gửi: " + u.getUserId() + " -> Trạm nguồn: " + nearest.getNodeId());
                } else {
                    view.lblStatus.setText("Không tìm thấy trạm nguồn cho " + username);
                }
            } else {
                view.lblStatus.setText("Không tìm thấy user: " + username);
            }
        } catch (Exception ex) {
            view.lblStatus.setText("Lỗi tìm người gửi: " + ex.getMessage());
        }
    }

    private void onDestinationUsernameSelected(String username) {
        if (username == null || username.isBlank()) return;
        if (userRepo == null || nodeService == null) return;
        
        try {
            var userOpt = userRepo.findByUserId(username);
            if (userOpt.isPresent()) {
                UserInfo u = userOpt.get();
                
                Optional<NodeInfo> nearestOptional = nodeService.getNearestNode(u.getUserId());
                NodeInfo nearest = nearestOptional.orElse(null);
                if (nearest != null) {
                    view.lblStatus.setText("Người nhận: " + u.getUserId() + " -> Trạm đích: " + nearest.getNodeId());
                } else {
                    view.lblStatus.setText("Không tìm thấy trạm đích cho " + username);
                }
            } else {
                view.lblStatus.setText("Không tìm thấy user: " + username);
            }
        } catch (Exception ex) {
            view.lblStatus.setText("Lỗi tìm người nhận: " + ex.getMessage());
        }
    }

    private void onPacketReceived(Packet p) {
        System.out.println("📥 CLIENT RECEIVED: ID=" + p.getPacketId() + ", useRL=" + p.isUseRL());
        
        Platform.runLater(() -> {
            receivedItems.add(0, p);
            view.lblStatus.setText("Đã nhận " + p.getPacketId() + (p.isUseRL() ? " (RL)" : " (non-RL)"));
        });
    }

    private Packet buildPacketFromForm() {
        Packet p = new Packet();

        p.setSourceUserId(view.cbSenderUsername.getValue());
        p.setDestinationUserId(view.cbDestinationUsername.getValue());

        if (nodeService != null) {
            if (p.getSourceUserId() != null) {
                Optional<NodeInfo> sourceNodeOpt = nodeService.getNearestNode(p.getSourceUserId());
                if (sourceNodeOpt.isEmpty()) {
                    System.err.println("Warning: Could not find source node for user " + p.getSourceUserId());
                }
                else {
                    NodeInfo sourceNode = sourceNodeOpt.get();
                    p.setStationSource(sourceNode.getNodeId());
                }
            }
            if (p.getDestinationUserId() != null) {
                Optional<NodeInfo> destNodeOpt = nodeService.getNearestNode(p.getDestinationUserId());
                if (destNodeOpt.isEmpty()) {
                    System.err.println("Warning: Could not find destination node for user " + p.getDestinationUserId());
                }
                else {
                    NodeInfo destNode = destNodeOpt.get();
                    p.setStationDest(destNode.getNodeId());
                }
            }
        }
    
        
        String payload = view.taPayload.getText();
        if (payload != null && !payload.isBlank()) {
            byte[] bytes = payload.getBytes(StandardCharsets.UTF_8);
            String base64 = Base64.getEncoder().encodeToString(bytes);
            p.setPayloadDataBase64(base64);
            
            try {
                String sizeStr = view.tfPayloadSizeByte.getText().trim();
                p.setPayloadSizeByte(sizeStr.isEmpty() ? bytes.length : Integer.parseInt(sizeStr));
            } catch (Exception ex) {
                p.setPayloadSizeByte(bytes.length);
            }
        } else {
            try {
                p.setPayloadSizeByte(Integer.parseInt(view.tfPayloadSizeByte.getText().trim()));
            } catch (Exception ignored) {}
        }

        ServiceType selected = view.cbServiceType.getValue();
        if (selected != null) {
            try {
                ServiceQoS qosProfile = QoSProfileFactory.getQosProfile(selected);
                p.setServiceQoS(qosProfile);
                p.setPriorityLevel(qosProfile.defaultPriority());
                p.setMaxAcceptableLatencyMs(qosProfile.maxLatencyMs());
                p.setMaxAcceptableLossRate(qosProfile.maxLossRate());
            } catch (Exception ignored) {}
        }

        p.setTTL(30);
        return p;
    }
} 