package com.chatapp.auth.model.entities;

public class User {
    private String email;
    private String idToken;

    public User(String email, String idToken) {
        this.email = email;
        this.idToken = idToken;
    }

    public String getEmail() {
        return email;
    }

    public String getIdToken() {
        return idToken;
    }
}
