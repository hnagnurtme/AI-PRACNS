package com.chatapp.chatapp.model.services;

import com.chatapp.chatapp.model.entities.Packet;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.net.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SocketChatServer {
    private static final int PORT = 8888;
    private ServerSocket serverSocket;
    private ExecutorService executorService;
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    // Map: email -> ClientHandler
    private final Map<String, ClientHandler> connectedClients = new ConcurrentHashMap<>();
    private volatile boolean running = false;
    
    public SocketChatServer() {
        this.executorService = Executors.newCachedThreadPool();
    }
    
    public void start() {
        try {
            serverSocket = new ServerSocket(PORT);
            running = true;
            
            System.out.println("🚀 Socket Chat Server started on port " + PORT);
            
            while (running) {
                try {
                    Socket clientSocket = serverSocket.accept();
                    ClientHandler handler = new ClientHandler(clientSocket);
                    executorService.submit(handler);
                    
                    System.out.println("👤 New client connected: " + clientSocket.getInetAddress());
                } catch (IOException e) {
                    if (running) {
                        System.err.println("Error accepting client: " + e.getMessage());
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Server error: " + e.getMessage());
        }
    }
    
    public void stop() {
        running = false;
        try {
            if (serverSocket != null) {
                serverSocket.close();
            }
            executorService.shutdown();
        } catch (IOException e) {
            System.err.println("Error stopping server: " + e.getMessage());
        }
    }
    
    // Broadcast message to specific user
    private void sendMessageToUser(String targetEmail, ChatMessage message) {
        ClientHandler targetClient = connectedClients.get(targetEmail);
        if (targetClient != null) {
            targetClient.sendMessage(message);
        } else {
            System.out.println("🔍 User " + targetEmail + " is not online");
        }
    }
    
    // Broadcast to all connected clients
    private void broadcastMessage(ChatMessage message) {
        for (ClientHandler client : connectedClients.values()) {
            client.sendMessage(message);
        }
    }
    
    // Client handler class
    private class ClientHandler implements Runnable {
        private Socket socket;
        private BufferedReader reader;
        private PrintWriter writer;
        private String userEmail;
        
        public ClientHandler(Socket socket) {
            this.socket = socket;
            try {
                reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                writer = new PrintWriter(socket.getOutputStream(), true);
            } catch (IOException e) {
                System.err.println("Error setting up client handler: " + e.getMessage());
            }
        }
        
        @Override
        public void run() {
            try {
                String inputLine;
                while ((inputLine = reader.readLine()) != null) {
                    handleMessage(inputLine);
                }
            } catch (IOException e) {
                System.out.println("Client disconnected: " + userEmail);
            } finally {
                cleanup();
            }
        }
        
        private void handleMessage(String messageJson) {
            try {
                ChatMessage message = objectMapper.readValue(messageJson, ChatMessage.class);
                
                switch (message.getType()) {
                    case "REGISTER":
                        registerUser(message.getSenderEmail());
                        break;
                        
                    case "CHAT":
                        handleChatMessage(message);
                        break;
                        
                    case "USER_SEARCH":
                        handleUserSearch(message);
                        break;
                        
                    default:
                        System.out.println("Unknown message type: " + message.getType());
                }
                
            } catch (Exception e) {
                System.err.println("Error handling message: " + e.getMessage());
            }
        }
        
        private void registerUser(String email) {
            this.userEmail = email;
            connectedClients.put(email, this);
            
            // Send confirmation
            ChatMessage response = new ChatMessage();
            response.setType("REGISTER_SUCCESS");
            response.setContent("Connected to chat server");
            sendMessage(response);
            
            System.out.println("✅ User registered: " + email);
        }
        
        private void handleChatMessage(ChatMessage message) {
            // Save to file for persistence
            saveMessageToFile(message);
            
            // Forward to recipient
            sendMessageToUser(message.getReceiverEmail(), message);
            
            // Send delivery confirmation to sender
            ChatMessage confirmation = new ChatMessage();
            confirmation.setType("DELIVERY_CONFIRMATION");
            confirmation.setMessageId(message.getMessageId());
            sendMessage(confirmation);
            
            System.out.println("📨 Message forwarded: " + message.getSenderEmail() + " -> " + message.getReceiverEmail());
        }
        
        private void handleUserSearch(ChatMessage message) {
            String searchEmail = message.getContent();
            boolean userOnline = connectedClients.containsKey(searchEmail);
            
            ChatMessage response = new ChatMessage();
            response.setType("USER_SEARCH_RESULT");
            response.setContent(searchEmail);
            response.setSenderEmail(userOnline ? "ONLINE" : "OFFLINE");
            sendMessage(response);
        }
        
        public void sendMessage(ChatMessage message) {
            try {
                String json = objectMapper.writeValueAsString(message);
                writer.println(json);
            } catch (Exception e) {
                System.err.println("Error sending message: " + e.getMessage());
            }
        }
        
        private void cleanup() {
            try {
                if (userEmail != null) {
                    connectedClients.remove(userEmail);
                }
                socket.close();
            } catch (IOException e) {
                System.err.println("Error during cleanup: " + e.getMessage());
            }
        }
    }
    
    private void saveMessageToFile(ChatMessage message) {
        try {
            String conversationId = generateConversationId(message.getSenderEmail(), message.getReceiverEmail());
            String fileName = "chat_history/" + conversationId + ".txt";
            
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
            String entry = String.format("[%s] %s -> %s: %s | ID: %s%n",
                timestamp, message.getSenderEmail(), message.getReceiverEmail(), 
                message.getContent(), message.getMessageId());
            
            java.nio.file.Files.write(java.nio.file.Paths.get(fileName), entry.getBytes(), 
                java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
                
        } catch (Exception e) {
            System.err.println("Error saving message to file: " + e.getMessage());
        }
    }
    
    private String generateConversationId(String user1, String user2) {
        return user1.compareTo(user2) < 0 ? user1 + "_" + user2 : user2 + "_" + user1;
    }
    
    // Message class for JSON serialization
    public static class ChatMessage {
        private String messageId = UUID.randomUUID().toString();
        private String type;
        private String senderEmail;
        private String receiverEmail;
        private String content;
        private long timestamp = System.currentTimeMillis();
        
        // Getters and setters
        public String getMessageId() { return messageId; }
        public void setMessageId(String messageId) { this.messageId = messageId; }
        
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        
        public String getSenderEmail() { return senderEmail; }
        public void setSenderEmail(String senderEmail) { this.senderEmail = senderEmail; }
        
        public String getReceiverEmail() { return receiverEmail; }
        public void setReceiverEmail(String receiverEmail) { this.receiverEmail = receiverEmail; }
        
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        
        public long getTimestamp() { return timestamp; }
        public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
    }
    
    // Main method for standalone server
    public static void main(String[] args) {
        SocketChatServer server = new SocketChatServer();
        
        // Graceful shutdown
        Runtime.getRuntime().addShutdownHook(new Thread(server::stop));
        
        server.start();
    }
}