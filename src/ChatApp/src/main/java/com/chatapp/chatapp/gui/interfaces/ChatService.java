package com.chatapp.chatapp.gui.interfaces;

import java.util.List;

public interface ChatService {
    void sendMessage(String message, String recipient);
    void receiveMessage(String message, String sender);
    List<String> getAvailableUsers();
    void updateUserList(List<String> users);
    void clearChatHistory();
}
