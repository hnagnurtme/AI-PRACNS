package com.chatapp.auth.model.entities;

public class UserSession {
    private static User currentUser;
    
    public static void setCurrentUser(User user) {
        currentUser = user;
    }
    
    public static User getCurrentUser() {
        return currentUser;
    }
    
    public static boolean isLoggedIn() {
        return currentUser != null;
    }
    
    public static void logout() {
        currentUser = null;
    }
    
    public static String getCurrentUserId() {
        return currentUser != null ? currentUser.getEmail() : "Anonymous";
    }
    
    public static String getCurrentUsername() {
        return currentUser != null ? currentUser.getEmail() : "Guest";
    }
}