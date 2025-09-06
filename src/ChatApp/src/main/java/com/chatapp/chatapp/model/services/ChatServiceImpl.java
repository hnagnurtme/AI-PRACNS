package com.chatapp.chatapp.model.services;

import java.time.LocalTime;
import java.util.List;

import com.chatapp.chatapp.model.entities.Packet;
import com.chatapp.chatapp.model.interfaces.ChatService;

import javafx.scene.control.ComboBox;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;

public class ChatServiceImpl implements ChatService {
    private final TextArea chatHistoryArea;
    private final TextField messageInputField;
    private final ComboBox<String> userSelectionCombo;
    
    public ChatServiceImpl(TextArea chatHistoryArea, TextField messageInputField, ComboBox<String> userSelectionCombo) {
        this.chatHistoryArea = chatHistoryArea;
        this.messageInputField = messageInputField;
        this.userSelectionCombo = userSelectionCombo;
        
        setupChatPanel();
        setupInitialUsers();
    }
    
    private void setupChatPanel() {
        if (chatHistoryArea != null) {
            chatHistoryArea.setEditable(false);
            chatHistoryArea.setWrapText(true);
        }
    }
    
    private void setupInitialUsers() {
        if (userSelectionCombo != null) {
            userSelectionCombo.getItems().addAll("Vo Luong", "Trung Anh", "BroadCast");
            userSelectionCombo.setValue("Vo Luong");
        }
    }
    
    @Override
    public void sendMessage(String message, String recipient) {
        // Tạo Packet object
        Packet packet;
        if ("BroadCast".equals(recipient)) {
            packet = Packet.createBroadcast("LocalUser", message);
        } else {
            packet = Packet.createMessage("LocalUser", recipient, message);
        }
        
        // Simulate routing
        simulateRouting(packet);
        
        // Log Packet object ra console
        logPacketObject(packet);
        
        // Hiển thị tin nhắn trong chat
        displayMessage(packet);
    }
    
    @Override
    public void receiveMessage(String message, String sender) {
        // Tạo Packet cho tin nhắn nhận
        Packet packet = Packet.createMessage(sender, "LocalUser", message);
        
        // Log và hiển thị
        logPacketObject(packet);
        displayReceivedMessage(packet);
    }
    
    // Simulate routing đơn giản
    private void simulateRouting(Packet packet) {
        // Thêm path
        packet.getPathHistory().add("LocalNode");
        packet.getPathHistory().add("Starlink");
        packet.getPathHistory().add("DestinationNode");
        
        // Set metrics
        packet.setCurrentNode("DestinationNode");
        packet.setNextHop("DestinationNode");
        packet.setDelayMs(Math.random() * 200 + 50); // 50-250ms
        packet.setLossRate(Math.random() * 0.01);    // 0-1%
    }
    
    // Log Packet object chi tiết
    private void logPacketObject(Packet packet) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("📦 PACKET OBJECT");
        System.out.println("=".repeat(60));
        System.out.println("ID: " + packet.getPacketId());
        System.out.println("From: " + packet.getSourceUserId());
        System.out.println("To: " + packet.getDestinationUserId());
        System.out.println("Message: \"" + packet.getMessage() + "\"");
        System.out.println("Timestamp: " + packet.getTimestamp());
        System.out.println("Payload Size: " + packet.getPayloadSize() + " bytes");
        System.out.println("TTL: " + packet.getTTL());
        System.out.println("Current Node: " + packet.getCurrentNode());
        System.out.println("Next Hop: " + packet.getNextHop());
        System.out.println("Path: " + String.join(" -> ", packet.getPathHistory()));
        System.out.println("Delay: " + String.format("%.2f ms", packet.getDelayMs()));
        System.out.println("Loss Rate: " + String.format("%.4f%%", packet.getLossRate() * 100));
        System.out.println("Retry Count: " + packet.getRetryCount());
        System.out.println("Priority: " + packet.getPriority());
        System.out.println("Dropped: " + packet.isDropped());
        System.out.println("toString(): " + packet.toString());
        System.out.println("=".repeat(60) + "\n");
    }
    
    // Hiển thị tin nhắn gửi
    private void displayMessage(Packet packet) {
        if (chatHistoryArea != null) {
            String timeStr = LocalTime.now().toString();
            String displayText = String.format("[%s] To %s: %s [%s]%n", 
                timeStr, 
                packet.getDestinationUserId(), 
                packet.getMessage(),
                packet.getPacketId()
            );
            chatHistoryArea.appendText(displayText);
            chatHistoryArea.setScrollTop(Double.MAX_VALUE);
        }
    }
    
    // Hiển thị tin nhắn nhận
    private void displayReceivedMessage(Packet packet) {
        if (chatHistoryArea != null) {
            String timeStr = LocalTime.now().toString();
            String displayText = String.format("[%s] From %s: %s [%s]%n", 
                timeStr,
                packet.getSourceUserId(),
                packet.getMessage(),
                packet.getPacketId()
            );
            chatHistoryArea.appendText(displayText);
            chatHistoryArea.setScrollTop(Double.MAX_VALUE);
        }
    }
    
    @Override
    public List<String> getAvailableUsers() {
        return List.of("Vo Luong", "Trung Anh", "BroadCast");
    }
    
    @Override
    public void updateUserList(List<String> users) {
        if (userSelectionCombo != null) {
            userSelectionCombo.getItems().clear();
            userSelectionCombo.getItems().addAll(users);
            if (!users.isEmpty()) {
                userSelectionCombo.setValue(users.get(0));
            }
        }
    }
    
    @Override
    public void clearChatHistory() {
        if (chatHistoryArea != null) {
            chatHistoryArea.clear();
        }
    }
    
    // Helper method for controller
    public void sendCurrentMessage() {
        if (messageInputField != null && userSelectionCombo != null) {
            String message = messageInputField.getText().trim();
            if (!message.isEmpty()) {
                String recipient = userSelectionCombo.getValue();
                sendMessage(message, recipient);
                messageInputField.clear();
            }
        }
    }
}