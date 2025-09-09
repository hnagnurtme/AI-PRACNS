package com.chatapp.chatapp.model.interfaces;

import java.util.Map;

public interface IChatService {
    void sendMessage(String message);
    Map<String, Object> searchUserByEmail(String email);
    void sendCurrentMessage();
    void clearChatHistory();
    
    // Method for search command
    void searchAndAddUser(String email);
}
