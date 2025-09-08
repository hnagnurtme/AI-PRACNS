package com.chatapp.chatapp.model.services;

import java.time.LocalTime;
import java.util.Map;
import java.util.List;

import com.chatapp.auth.model.entities.UserSession;
import com.chatapp.chatapp.config.FirebaseConfig;
import com.chatapp.chatapp.model.entities.Packet;
import com.chatapp.chatapp.model.interfaces.IChatService;
import com.google.cloud.firestore.*;

import javafx.application.Platform;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;

public class ChatService implements IChatService {
    private final TextArea chatHistoryArea;
    private final TextField messageInputField;
    private final ComboBox<String> userSelectionCombo;
    private final ChatFileService fileService;
    
    private String currentConversationId;
    private String currentReceiverEmail;
    
    public ChatService(TextArea chatHistoryArea, TextField messageInputField, ComboBox<String> userSelectionCombo) {
        this.chatHistoryArea = chatHistoryArea;
        this.messageInputField = messageInputField;
        this.userSelectionCombo = userSelectionCombo;
        this.fileService = new ChatFileService();
        
        setupChatPanel();
        setupInitialUsers();
    }
    
    private void setupChatPanel() {
        if (chatHistoryArea != null) {
            chatHistoryArea.setEditable(false);
            chatHistoryArea.setWrapText(true);
            chatHistoryArea.appendText("💬 Welcome to SAGIN Chat!\n");
            chatHistoryArea.appendText("📝 Type '/search email@example.com' to find and start chatting with someone.\n\n");
        }
    }
    
    private void setupInitialUsers() {
        if (userSelectionCombo != null) {
            userSelectionCombo.getItems().clear();
            userSelectionCombo.getItems().add("Select a user to chat...");
            userSelectionCombo.setValue("Select a user to chat...");
        }
    }
    
    @Override
    public void sendMessage(String message, String recipient) {
        String currentUser = UserSession.getCurrentUserId();
        if (currentUser == null || currentUser.equals("Anonymous")) {
            System.err.println("User not logged in");
            return;
        }
        
        // Check for search command
        if (message.startsWith("/search ")) {
            String email = message.substring(8).trim();
            searchAndAddUser(email);
            return;
        }
        
        if (currentReceiverEmail == null) {
            if (chatHistoryArea != null) {
                chatHistoryArea.appendText("⚠️ Use '/search email@example.com' to find someone to chat with!\n");
            }
            return;
        }
        
        // Tạo packet
        Packet packet = Packet.createMessage(currentUser, currentReceiverEmail, message);
        
        // Simulate routing
        simulateRouting(packet);
        
        // Log packet
        logPacketObject(packet);
        
        // Lưu vào file
        fileService.saveChatMessage(packet);
        
        // Hiển thị tin nhắn của mình
        if (chatHistoryArea != null) {
            String timeStr = LocalTime.now().toString();
            String displayText = String.format("[%s] You: %s", timeStr, message);
            chatHistoryArea.appendText(displayText + "\n");
            chatHistoryArea.setScrollTop(Double.MAX_VALUE);
        }
    }
    
    public void searchAndAddUser(String email) {
        if (chatHistoryArea != null) {
            chatHistoryArea.appendText("🔍 Searching for: " + email + "\n");
        }
        
        Map<String, Object> userData = searchUserByEmail(email);
        
        if (userData != null) {
            String foundEmail = (String) userData.get("email");
            String displayName = (String) userData.getOrDefault("displayName", foundEmail);
            
            // Update receiver
            currentReceiverEmail = foundEmail;
            
            // Update UI
            if (userSelectionCombo != null) {
                String displayText = displayName + " (" + foundEmail + ")";
                userSelectionCombo.getItems().clear();
                userSelectionCombo.getItems().add(displayText);
                userSelectionCombo.setValue(displayText);
            }
            
            // Generate conversation ID
            String currentUser = UserSession.getCurrentUserId();
            currentConversationId = generateConversationId(currentUser, foundEmail);
            
            // Load chat history
            loadAndDisplayChatHistory();
            
            // Start watching for new messages
            startWatchingConversation();
            
            if (chatHistoryArea != null) {
                chatHistoryArea.appendText("✅ Connected to chat with: " + displayName + "\n");
                chatHistoryArea.appendText("=".repeat(50) + "\n");
            }
            
        } else {
            if (chatHistoryArea != null) {
                chatHistoryArea.appendText("❌ User not found: " + email + "\n");
            }
        }
    }
    
    private void loadAndDisplayChatHistory() {
        if (currentConversationId != null && chatHistoryArea != null) {
            List<String> history = fileService.loadChatHistory(currentConversationId, 20);
            
            if (!history.isEmpty()) {
                chatHistoryArea.appendText("📚 Chat History:\n");
                for (String message : history) {
                    chatHistoryArea.appendText(message + "\n");
                }
                chatHistoryArea.appendText("--- New Messages ---\n");
            }
            
            chatHistoryArea.setScrollTop(Double.MAX_VALUE);
        }
    }
    
    private void startWatchingConversation() {
        if (currentConversationId != null) {
            String currentUser = UserSession.getCurrentUserId();
            
            fileService.startWatchingConversation(currentConversationId, new ChatFileService.ChatFileListener() {
                @Override
                public void onNewMessage(String message) {
                    Platform.runLater(() -> {
                        // Only show messages from others (not our own)
                        if (!message.contains(currentUser + " ->")) {
                            if (chatHistoryArea != null) {
                                chatHistoryArea.appendText("📨 " + message + "\n");
                                chatHistoryArea.setScrollTop(Double.MAX_VALUE);
                            }
                        }
                    });
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
                QueryDocumentSnapshot doc = snapshot.getDocuments().get(0);
                Map<String, Object> userData = doc.getData();
                userData.put("uid", doc.getId());
                return userData;
            } else {
                return null;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    
    @Override
    public void sendCurrentMessage() {
        if (messageInputField != null) {
            String message = messageInputField.getText().trim();
            if (!message.isEmpty()) {
                sendMessage(message, currentReceiverEmail);
                messageInputField.clear();
            }
        }
    }
    
    @Override
    public void clearChatHistory() {
        if (chatHistoryArea != null) {
            chatHistoryArea.clear();
            setupChatPanel(); // Show welcome message again
        }
    }
    
    // Helper methods
    private String generateConversationId(String user1, String user2) {
        return user1.compareTo(user2) < 0 ? user1 + "_" + user2 : user2 + "_" + user1;
    }
    
    private void simulateRouting(Packet packet) {
        packet.getPathHistory().add("LocalNode");
        packet.getPathHistory().add("Starlink");
        packet.getPathHistory().add("DestinationNode");
        
        packet.setCurrentNode("DestinationNode");
        packet.setNextHop("DestinationNode");
        packet.setDelayMs(Math.random() * 200 + 50);
        packet.setLossRate(Math.random() * 0.01);
    }
    
    private void logPacketObject(Packet packet) {
        System.out.println("\n" + "=".repeat(50));
        System.out.println("📦 CHAT PACKET");
        System.out.println("From: " + packet.getSourceUserId());
        System.out.println("To: " + packet.getDestinationUserId());
        System.out.println("Message: \"" + packet.getMessage() + "\"");
        System.out.println("=".repeat(50) + "\n");
    }
    
    // Cleanup
    public void shutdown() {
        if (currentConversationId != null) {
            fileService.stopWatchingConversation(currentConversationId);
        }
    }
}
