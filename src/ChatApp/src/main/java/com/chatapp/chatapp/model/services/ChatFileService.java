package com.chatapp.chatapp.model.services;

import com.chatapp.chatapp.model.entities.Packet;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

public class ChatFileService {
    // Sử dụng absolute path cho Docker
    private static final String CHAT_DIR = getChatDirectory();
    private static final String CHAT_EXTENSION = ".txt";
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    private Map<String, WatchService> fileWatchers = new HashMap<>();
    private Map<String, List<ChatFileListener>> listeners = new HashMap<>();
    
    /**
     * Determine chat directory based on environment
     */
    private static String getChatDirectory() {
        // Trong Docker container
        if (Files.exists(Paths.get("/app/chat_history"))) {
            return "/app/chat_history";
        }
        // Local development
        return "chat_history";
    }
    
    public ChatFileService() {
        try {
            Files.createDirectories(Paths.get(CHAT_DIR));
            System.out.println("📁 Chat directory: " + Paths.get(CHAT_DIR).toAbsolutePath());
        } catch (IOException e) {
            System.err.println("Failed to create chat directory: " + e.getMessage());
        }
    }
    
    /**
     * Lưu tin nhắn vào file
     */
    public void saveChatMessage(Packet packet) {
        String conversationId = generateConversationId(packet.getSourceUserId(), packet.getDestinationUserId());
        String fileName = getChatFileName(conversationId);
        
        try {
            // Format tin nhắn
            String messageEntry = formatMessageEntry(packet);
            
            // Append vào file
            Files.write(
                Paths.get(fileName), 
                (messageEntry + "\n").getBytes(), 
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND
            );
            
            System.out.println("💾 Message saved to: " + fileName);
            
        } catch (IOException e) {
            System.err.println("Failed to save message: " + e.getMessage());
        }
    }
    
    /**
     * Load lịch sử chat từ file
     */
    public List<String> loadChatHistory(String conversationId, int maxLines) {
        String fileName = getChatFileName(conversationId);
        List<String> messages = new ArrayList<>();
        
        try {
            if (!Files.exists(Paths.get(fileName))) {
                return messages; // File chưa tồn tại
            }
            
            List<String> allLines = Files.readAllLines(Paths.get(fileName));
            
            // Lấy maxLines cuối cùng
            int startIndex = Math.max(0, allLines.size() - maxLines);
            messages = allLines.subList(startIndex, allLines.size());
            
        } catch (IOException e) {
            System.err.println("Failed to load chat history: " + e.getMessage());
        }
        
        return messages;
    }

    public java.util.List<String> loadChatHistory(String user1, String user2) {
        java.util.List<String> messages = new java.util.ArrayList<>();
        String conversationId = generateConversationId(user1, user2);
        String fileName = getChatFileName(conversationId);
        
        try {
            java.nio.file.Path filePath = java.nio.file.Paths.get(fileName);
            if (java.nio.file.Files.exists(filePath)) {
                messages = java.nio.file.Files.readAllLines(filePath);
            }
        } catch (Exception e) {
            System.err.println("Error loading chat history: " + e.getMessage());
        }
        
        return messages;
    }
    
    /**
     * Bắt đầu watch file changes cho real-time
     */
    public void startWatchingConversation(String conversationId, ChatFileListener listener) {
        String fileName = getChatFileName(conversationId);
        
        try {
            // Tạo file nếu chưa có
            Path filePath = Paths.get(fileName);
            if (!Files.exists(filePath)) {
                Files.createFile(filePath);
            }
            
            // Stop existing watcher
            stopWatchingConversation(conversationId);
            
            // Create new watcher
            WatchService watchService = FileSystems.getDefault().newWatchService();
            Path directory = filePath.getParent();
            
            directory.register(watchService, StandardWatchEventKinds.ENTRY_MODIFY);
            
            // Add listener
            listeners.computeIfAbsent(conversationId, k -> new CopyOnWriteArrayList<>()).add(listener);
            fileWatchers.put(conversationId, watchService);
            
            // Start watching in separate thread
            Thread watchThread = new Thread(() -> watchFileChanges(conversationId, watchService, filePath));
            watchThread.setDaemon(true);
            watchThread.start();
            
            System.out.println("👁️ Started watching file: " + fileName);
            
        } catch (IOException e) {
            System.err.println("Failed to start file watcher: " + e.getMessage());
        }
    }
    
    /**
     * Dừng watch file
     */
    public void stopWatchingConversation(String conversationId) {
        WatchService watchService = fileWatchers.remove(conversationId);
        if (watchService != null) {
            try {
                watchService.close();
            } catch (IOException e) {
                System.err.println("Error closing file watcher: " + e.getMessage());
            }
        }
        listeners.remove(conversationId);
    }
    
    // Helper methods
    private String getChatFileName(String conversationId) {
        return CHAT_DIR + File.separator + conversationId + CHAT_EXTENSION;
    }
    
    private String formatMessageEntry(Packet packet) {
        LocalDateTime now = LocalDateTime.now();
        String timestamp = now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        
        return String.format("[%s] %s -> %s: %s | PacketID: %s",
            timestamp,
            packet.getSourceUserId(),
            packet.getDestinationUserId(),
            packet.getMessage(),
            packet.getPacketId()
        );
    }
    
    private void watchFileChanges(String conversationId, WatchService watchService, Path filePath) {
        try {
            WatchKey key;
            while ((key = watchService.take()) != null) {
                for (WatchEvent<?> event : key.pollEvents()) {
                    if (event.kind() == StandardWatchEventKinds.ENTRY_MODIFY) {
                        Path changed = (Path) event.context();
                        if (changed.equals(filePath.getFileName())) {
                            // File đã thay đổi, đọc dòng cuối
                            String lastLine = getLastLineFromFile(filePath.toString());
                            if (lastLine != null && !lastLine.trim().isEmpty()) {
                                notifyListeners(conversationId, lastLine);
                            }
                        }
                    }
                }
                key.reset();
            }
        } catch (InterruptedException e) {
            System.out.println("File watcher interrupted for: " + conversationId);
        } catch (Exception e) {
            System.err.println("Error in file watcher: " + e.getMessage());
        }
    }
    
    private String getLastLineFromFile(String fileName) {
        try (RandomAccessFile file = new RandomAccessFile(fileName, "r")) {
            long fileLength = file.length() - 1;
            if (fileLength < 0) return null;
            
            StringBuilder sb = new StringBuilder();
            for (long pointer = fileLength; pointer >= 0; pointer--) {
                file.seek(pointer);
                char c = (char) file.read();
                if (c == '\n' && sb.length() > 0) {
                    break;
                }
                sb.append(c);
            }
            
            return sb.reverse().toString().trim();
        } catch (IOException e) {
            return null;
        }
    }
    
    private void notifyListeners(String conversationId, String newMessage) {
        List<ChatFileListener> conversationListeners = listeners.get(conversationId);
        if (conversationListeners != null) {
            for (ChatFileListener listener : conversationListeners) {
                try {
                    listener.onNewMessage(newMessage);
                } catch (Exception e) {
                    System.err.println("Error notifying listener: " + e.getMessage());
                }
            }
        }
    }
    
    public interface ChatFileListener {
        void onNewMessage(String message);
    }

    // Thêm method helper
    private String generateConversationId(String user1, String user2) {
        return user1.compareTo(user2) < 0 ? user1 + "_" + user2 : user2 + "_" + user1;
    }
}
