package com.chatapp.chatapp.model.services;

import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.application.Platform;

import java.io.*;
import java.net.*;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;

public class SocketChatClient {
    // Đọc từ environment variables
    private static final String SERVER_HOST = System.getenv("SOCKET_SERVER_HOST") != null 
        ? System.getenv("SOCKET_SERVER_HOST") 
        : "localhost";
    private static final int SERVER_PORT = System.getenv("SOCKET_SERVER_PORT") != null 
        ? Integer.parseInt(System.getenv("SOCKET_SERVER_PORT")) 
        : 8888;
    
    private Socket socket;
    private BufferedReader reader;
    private PrintWriter writer;
    private ObjectMapper objectMapper = new ObjectMapper();
    private Consumer<SocketChatServer.ChatMessage> messageHandler;
    private volatile boolean connected = false;
    
    private String currentUserEmail;
    
    public CompletableFuture<Boolean> connect(String userEmail) {
        this.currentUserEmail = userEmail;
        return CompletableFuture.supplyAsync(() -> {
            try {
                System.out.println("🔌 Trying to connect to " + SERVER_HOST + ":" + SERVER_PORT);
                
                socket = new Socket();
                socket.connect(new InetSocketAddress(SERVER_HOST, SERVER_PORT), 5000); // 5 second timeout
                
                reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                writer = new PrintWriter(socket.getOutputStream(), true);
                connected = true;
                
                System.out.println("🔗 Socket connected, registering user: " + userEmail);
                
                // Register user
                SocketChatServer.ChatMessage registerMessage = new SocketChatServer.ChatMessage();
                registerMessage.setType("REGISTER");
                registerMessage.setSenderEmail(userEmail);
                sendMessage(registerMessage);
                
                // Start listening for incoming messages
                startListening();
                
                System.out.println("✅ Successfully connected to chat server as: " + userEmail);
                return true;
                
            } catch (ConnectException e) {
                System.err.println("❌ Connection refused - Server is not running on port " + SERVER_PORT);
                System.err.println("💡 Start server: mvn exec:java -Dexec.mainClass=\"com.chatapp.chatapp.model.services.SocketChatServer\"");
                return false;
            } catch (SocketTimeoutException e) {
                System.err.println("❌ Connection timeout - Server not responding");
                return false;
            } catch (IOException e) {
                System.err.println("❌ Failed to connect to server: " + e.getMessage());
                return false;
            } catch (Exception e) {
                System.err.println("❌ Unexpected error: " + e.getMessage());
                e.printStackTrace();
                return false;
            }
        });
    }
    
    public void setMessageHandler(Consumer<SocketChatServer.ChatMessage> handler) {
        this.messageHandler = handler;
    }
    
    public void sendChatMessage(String receiverEmail, String content) {
        if (!isConnected()) {
            System.err.println("❌ Cannot send message - not connected to server");
            return;
        }
        
        SocketChatServer.ChatMessage message = new SocketChatServer.ChatMessage();
        message.setType("CHAT");
        message.setSenderEmail(this.currentUserEmail);
        message.setReceiverEmail(receiverEmail);
        message.setContent(content);
        
        System.out.println("📤 Sending chat: " + currentUserEmail + " -> " + receiverEmail + ": " + content);
        sendMessage(message);
    }
    
    public void searchUser(String email) {
        SocketChatServer.ChatMessage message = new SocketChatServer.ChatMessage();
        message.setType("USER_SEARCH");
        message.setContent(email);
        
        sendMessage(message);
    }
    
    private void sendMessage(SocketChatServer.ChatMessage message) {
        if (!connected || writer == null) {
            System.err.println("❌ Not connected to server");
            return;
        }
        
        try {
            String json = objectMapper.writeValueAsString(message);
            writer.println(json);
            System.out.println("📡 Sent JSON: " + json);
        } catch (Exception e) {
            System.err.println("❌ Error sending message: " + e.getMessage());
        }
    }
    
    private void startListening() {
        Thread listenerThread = new Thread(() -> {
            try {
                String inputLine;
                while (connected && (inputLine = reader.readLine()) != null) {
                    System.out.println("📥 Received: " + inputLine);
                    handleIncomingMessage(inputLine);
                }
            } catch (IOException e) {
                if (connected) {
                    System.err.println("❌ Connection lost: " + e.getMessage());
                    connected = false;
                }
            }
        });
        listenerThread.setDaemon(true);
        listenerThread.start();
        System.out.println("👂 Started listening for messages");
    }
    
    private void handleIncomingMessage(String messageJson) {
        try {
            SocketChatServer.ChatMessage message = objectMapper.readValue(messageJson, SocketChatServer.ChatMessage.class);
            System.out.println("📨 Parsed message type: " + message.getType());
            
            // Handle on JavaFX thread if needed
            if (messageHandler != null) {
                Platform.runLater(() -> messageHandler.accept(message));
            }
            
        } catch (Exception e) {
            System.err.println("❌ Error handling incoming message: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public void disconnect() {
        connected = false;
        try {
            if (socket != null) {
                socket.close();
            }
            System.out.println("🔌 Disconnected from server");
        } catch (IOException e) {
            System.err.println("Error disconnecting: " + e.getMessage());
        }
    }
    
    public boolean isConnected() {
        return connected && socket != null && !socket.isClosed();
    }
}