package com.chatapp.chatapp.model.services;

import java.time.LocalTime;
import java.util.Map;

import com.chatapp.auth.model.entities.UserSession;
import com.chatapp.chatapp.config.FirebaseConfig;
import com.chatapp.chatapp.model.interfaces.IChatService;
import com.chatapp.chatapp.model.entities.Packet;
import com.google.cloud.firestore.*;

import javafx.application.Platform;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;

import java.net.Socket;

import com.chatapp.chatapp.helpers.GeoLocationService;
import com.chatapp.chatapp.model.entities.Node;

public class ChatService implements IChatService {
    private final TextArea chatHistoryArea;
    private final TextField messageInputField;
    private final ComboBox<String> nodeSelector;
    
    private SocketChatClient socketClient;
    private ChatFileService fileService;
    private String currentUserEmail;
    private String currentChatPartner;
    
    private static SocketChatServer embeddedServer;
    private static boolean serverStarted = false;
    
    public ChatService(TextArea chatHistoryArea, TextField messageInputField, ComboBox<String> nodeSelector) {
        this.chatHistoryArea = chatHistoryArea;
        this.messageInputField = messageInputField;
        this.nodeSelector = nodeSelector;
        
        this.socketClient = new SocketChatClient();
        this.fileService = new ChatFileService();
        
        // Setup message handler for incoming messages
        socketClient.setMessageHandler(message -> {
            Platform.runLater(() -> {
                handleIncomingMessage(message);
            });
        });
    }

    public void initialize() {
        currentUserEmail = UserSession.getCurrentUser().getEmail();
        
        // Start embedded server if not running
        startEmbeddedServerIfNeeded();
        
        // Small delay to let server start
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        // Connect to socket server
        chatHistoryArea.appendText("🔌 Connecting to chat server...\n");
        
        socketClient.connect(currentUserEmail).thenAccept(success -> {
            Platform.runLater(() -> {
                if (success) {
                    chatHistoryArea.appendText("✅ Connected to chat server!\n");
                    chatHistoryArea.appendText("💡 Type '/search email@example.com' to find a user\n");
                    chatHistoryArea.appendText("=".repeat(50) + "\n");
                } else {
                    chatHistoryArea.appendText("❌ Failed to connect to chat server\n");
                    chatHistoryArea.appendText("💡 You can still search users and view chat history\n");
                    chatHistoryArea.appendText("=".repeat(50) + "\n");
                }
            });
        });
    }

    private void handleIncomingMessage(SocketChatServer.ChatMessage message) {
        // 🆕 Log detailed packet information
        logIncomingPacket(message);
        
        switch (message.getType()) {
            case "REGISTER_SUCCESS":
                chatHistoryArea.appendText("🔗 " + message.getContent() + "\n");
                break;
                
            case "CHAT":
                // Display received chat message
                String senderEmail = message.getSenderEmail();
                String content = message.getContent();
                String timestamp = LocalTime.now().toString().substring(0, 8);
                
                String displayMessage = String.format("[%s] %s: %s\n", 
                    timestamp, senderEmail.split("@")[0], content);
                chatHistoryArea.appendText(displayMessage);
                
                // Save to file for history
                saveMessageToHistory(senderEmail, currentUserEmail, content);
                break;
                
            case "USER_SEARCH_RESULT":
                String searchedEmail = message.getContent();
                String status = message.getSenderEmail(); // ONLINE or OFFLINE
                chatHistoryArea.appendText("📶 " + searchedEmail + " is " + status + "\n");
                break;
                
            case "DELIVERY_CONFIRMATION":
                System.out.println("✅ Message delivered: " + message.getMessageId());
                break;
                
            default:
                System.out.println("Unknown message type: " + message.getType());
        }
    }

    // 🆕 Thêm method mới để log packet details
    private void logIncomingPacket(SocketChatServer.ChatMessage message) {
        System.out.println("=" + "=".repeat(60) + "=");
        System.out.println("📦 INCOMING PACKET LOG");
        System.out.println("=" + "=".repeat(60) + "=");
        System.out.println("🔍 Packet Type: " + message.getType());
        System.out.println("👤 Sender Email: " + message.getSenderEmail());
        System.out.println("📧 Receiver Email: " + message.getReceiverEmail());
        System.out.println("💬 Content: " + message.getContent());
        System.out.println("🆔 Message ID: " + message.getMessageId());
        System.out.println("⏰ Timestamp: " + java.time.LocalDateTime.now());
        System.out.println("📊 Content Length: " + (message.getContent() != null ? message.getContent().length() : 0) + " chars");
        
        // Additional packet simulation data
        System.out.println("🌐 Simulated Network Info:");
        System.out.println("   📍 Current Node: " + currentUserEmail);
        System.out.println("   📡 Source: " + message.getSenderEmail());
        System.out.println("   🎯 Destination: " + message.getReceiverEmail());
        System.out.println("   ⚡ Estimated Delay: " + String.format("%.2f ms", Math.random() * 200 + 50));
        System.out.println("   📦 Packet Size: " + estimatePacketSize(message) + " bytes");
        System.out.println("=" + "=".repeat(60) + "=");
    }

    // 🆕 Helper method để ước tính kích thước packet
    private int estimatePacketSize(SocketChatServer.ChatMessage message) {
        int headerSize = 64; // Estimated header size
        int contentSize = message.getContent() != null ? message.getContent().getBytes().length : 0;
        int senderSize = message.getSenderEmail() != null ? message.getSenderEmail().getBytes().length : 0;
        int receiverSize = message.getReceiverEmail() != null ? message.getReceiverEmail().getBytes().length : 0;
        int typeSize = message.getType() != null ? message.getType().getBytes().length : 0;
        
        return headerSize + contentSize + senderSize + receiverSize + typeSize;
    }

    @Override
    public void sendMessage(String message) {
        System.out.println("🔧 DEBUG - sendMessage called with: " + message);
        
        if (message.startsWith("/search ")) {
            String email = message.substring(8).trim();
            searchAndAddUser(email);
            return;
        }
        
        if (message.startsWith("/debug")) {
            showDebugInfo();
            return;
        }

        if (currentChatPartner == null) {
            chatHistoryArea.appendText("❌ Please search for a user first using '/search email@example.com'\n");
            return;
        }

        if (message.trim().isEmpty()) {
            System.out.println("❌ Empty message");
            return;
        }

        System.out.println("📤 Sending message: " + currentUserEmail + " -> " + currentChatPartner + ": " + message);

        // Send via socket (real-time)
        if (socketClient.isConnected()) {
            socketClient.sendChatMessage(currentChatPartner, message);
            System.out.println("✅ Message sent via socket");
        } else {
            chatHistoryArea.appendText("⚠️ Offline mode - message saved locally\n");
            System.out.println("⚠️ Socket not connected");
        }
        
        // Display in local chat
        String timestamp = LocalTime.now().toString().substring(0, 8);
        String displayMessage = String.format("[%s] You: %s\n", timestamp, message);
        chatHistoryArea.appendText(displayMessage);
        
        // Save to history file
        saveMessageToHistory(currentUserEmail, currentChatPartner, message);
        
        if (messageInputField != null) {
            messageInputField.clear();
        }

        // Tìm node gần nhất và hiển thị thông tin
        try {
            double[] userLocation = GeoLocationService.getCurrentLocation();
            double userLat = userLocation[0];
            double userLon = userLocation[1];

            System.out.printf("📍 Your location: (%.4f, %.4f)\n", userLat, userLon);
            UserSession.getCurrentUser().setLocation(userLat, userLon);

            NodeService nodeService = new NodeService(FirebaseConfig.getFirestore());

            Node nearestNode = nodeService.findNearestNode(UserSession.getCurrentUser());

            if (nearestNode != null) {
                System.out.printf("📡 Connected to:  %s  %s  %.4f  %.4f\n",
                        nearestNode.getNodeId(),
                        nearestNode.getNodeType(),
                        nearestNode.getLatitude(),
                        nearestNode.getLongitude());
            } else {
                chatHistoryArea.appendText("⚠️ No communication nodes available\n");
                System.out.println("⚠️ No communication nodes available");
            }
        } catch (Exception e) {
            System.err.println("Error finding nearest node: " + e.getMessage());
        }
    }

    @Override
    public void sendCurrentMessage() {
        String message = messageInputField.getText().trim();
        if (!message.isEmpty()) {
            sendMessage(message);
        }
    }

    @Override
    public void clearChatHistory() {
        if (chatHistoryArea != null) {
            chatHistoryArea.clear();
        }
    }

    @Override
    public void searchAndAddUser(String email) {
        chatHistoryArea.appendText("🔍 Searching for: " + email + "\n");
        
        // Search user in Firebase first
        Map<String, Object> userData = searchUserByEmail(email);
        
        if (userData != null) {
            currentChatPartner = email;
            String displayName = userData.getOrDefault("displayName", email.split("@")[0]).toString();
            
            chatHistoryArea.appendText("✅ Found user: " + displayName + " (" + email + ")\n");
            chatHistoryArea.appendText("💬 You can now start chatting!\n");
            
            // **UPDATE COMBOBOX**
            Platform.runLater(() -> {
                if (nodeSelector != null) {
                    // Clear and add the found user
                    nodeSelector.getItems().clear();
                    nodeSelector.getItems().add(displayName + " (" + email + ")");
                    nodeSelector.setValue(displayName + " (" + email + ")");
                    nodeSelector.setDisable(false);
                }
            });
            
            // Check if user is online via socket
            if (socketClient.isConnected()) {
                socketClient.searchUser(email);
            }
            
            // Load and display chat history
            loadAndDisplayChatHistory();
            
            chatHistoryArea.appendText("=".repeat(50) + "\n");
        } else {
            chatHistoryArea.appendText("❌ User not found in database: " + email + "\n");
            
            // Clear ComboBox if user not found
            Platform.runLater(() -> {
                if (nodeSelector != null) {
                    nodeSelector.getItems().clear();
                    nodeSelector.setDisable(true);
                }
            });
        }
    }

    @Override
    public Map<String, Object> searchUserByEmail(String email) {
        try {
            Firestore db = FirebaseConfig.getFirestore();
            QuerySnapshot snapshot = db.collection("users")
                .whereEqualTo("email", email)
                .get()
                .get();

            if (!snapshot.isEmpty()) {
                DocumentSnapshot document = snapshot.getDocuments().get(0);
                return document.getData();
            }
            return null;
        } catch (Exception e) {
            System.err.println("Error searching user: " + e.getMessage());
            return null;
        }
    }

    private void saveMessageToHistory(String senderEmail, String receiverEmail, String content) {
        try {
            // Create packet for file storage
            Packet packet = new Packet();
            packet.setSourceUserId(senderEmail);
            packet.setDestinationUserId(receiverEmail);
            packet.setMessage(content);
            packet.setTimestamp(System.currentTimeMillis());
            packet.setPacketId(java.util.UUID.randomUUID().toString());
            packet.setCurrentNode(currentUserEmail);
            packet.setNextHop(receiverEmail);
            packet.setDelayMs(Math.random() * 200 + 50); // 50-250ms
            packet.setLossRate(Math.random() * 0.01);    // 0-1%
            packet.setRetryCount(0);
            packet.setPriority(5);
            packet.setDropped(false);
            
            // 🆕 Log outgoing packet details khi save
            logOutgoingPacket(packet);
            
            // Save to file
            fileService.saveChatMessage(packet);
            
        } catch (Exception e) {
            System.err.println("Error saving message to history: " + e.getMessage());
        }
    }

    // 🆕 Thêm method để log outgoing packet
    private void logOutgoingPacket(Packet packet) {
        System.out.println("=" + "=".repeat(60) + "=");
        System.out.println("📤 OUTGOING PACKET LOG");
        System.out.println("=" + "=".repeat(60) + "=");
        System.out.println("🆔 Packet ID: " + packet.getPacketId());
        System.out.println("👤 Source User: " + packet.getSourceUserId());
        System.out.println("🎯 Destination User: " + packet.getDestinationUserId());
        System.out.println("💬 Message: " + packet.getMessage());
        System.out.println("📍 Current Node: " + packet.getCurrentNode());
        System.out.println("➡️ Next Hop: " + packet.getNextHop());
        System.out.println("⏱️ Delay: " + String.format("%.2f ms", packet.getDelayMs()));
        System.out.println("📉 Loss Rate: " + String.format("%.4f%%", packet.getLossRate() * 100));
        System.out.println("🔄 Retry Count: " + packet.getRetryCount());
        System.out.println("⭐ Priority: " + packet.getPriority());
        System.out.println("❌ Dropped: " + packet.isDropped());
        System.out.println("⏰ Timestamp: " + new java.util.Date(packet.getTimestamp()));
        System.out.println("=" + "=".repeat(60) + "=");
    }

    private void loadAndDisplayChatHistory() {
        if (currentChatPartner == null) return;
        
        try {
            // Load chat history from file
            java.util.List<String> messages = fileService.loadChatHistory(currentUserEmail, currentChatPartner);
            
            if (!messages.isEmpty()) {
                chatHistoryArea.appendText("📚 Loading chat history...\n");
                chatHistoryArea.appendText("-".repeat(30) + "\n");
                
                // Display last 20 messages
                int startIndex = Math.max(0, messages.size() - 20);
                for (int i = startIndex; i < messages.size(); i++) {
                    chatHistoryArea.appendText(messages.get(i) + "\n");
                }
                
                chatHistoryArea.appendText("-".repeat(30) + "\n");
                if (messages.size() > 20) {
                    chatHistoryArea.appendText("... (" + (messages.size() - 20) + " older messages)\n");
                }
            }
            
        } catch (Exception e) {
            System.err.println("Error loading chat history: " + e.getMessage());
        }
    }

    private void showDebugInfo() {
        chatHistoryArea.appendText("=== DEBUG INFO ===\n");
        chatHistoryArea.appendText("Current User: " + currentUserEmail + "\n");
        chatHistoryArea.appendText("Chat Partner: " + (currentChatPartner != null ? currentChatPartner : "None") + "\n");
        chatHistoryArea.appendText("Socket Connected: " + socketClient.isConnected() + "\n");
        chatHistoryArea.appendText("File Service: " + (fileService != null ? "Ready" : "Not Ready") + "\n");
        chatHistoryArea.appendText("================\n");
    }

    public void shutdown() {
        cleanup();
    }

    public void cleanup() {
        if (socketClient != null) {
            socketClient.disconnect();
        }
    }

    private void startEmbeddedServerIfNeeded() {
        if (!serverStarted) {
            synchronized (ChatService.class) {
                if (!serverStarted) {
                    try {
                        // Check if port is already in use
                        try (Socket testSocket = new Socket()) {
                            testSocket.connect(new java.net.InetSocketAddress("localhost", 8888), 1000);
                            System.out.println("📡 Server already running on port 8888");
                            serverStarted = true;
                            return;
                        } catch (Exception e) {
                            // Port not in use, start server
                        }
                        
                        embeddedServer = new SocketChatServer();
                        Thread serverThread = new Thread(() -> {
                            System.out.println("🚀 Starting embedded chat server...");
                            embeddedServer.start();
                        });
                        serverThread.setDaemon(true);
                        serverThread.start();
                        
                        serverStarted = true;
                        System.out.println("✅ Embedded server started");
                        
                    } catch (Exception e) {
                        System.err.println("❌ Failed to start embedded server: " + e.getMessage());
                    }
                }
            }
        }
    }
}
