package com.chatapp.auth.model.entities;

public class User {
    private String email;
    private String idToken;
    private double latitude;
    private double longitude;

    public User(String email, String idToken, double latitude, double longitude) {
        this.email = email;
        this.idToken = idToken;
        this.latitude = latitude;
        this.longitude = longitude;
    }

    public String getEmail() {
        return email;
    }

    public String getIdToken() {
        return idToken;
    }

    public double getLatitude() {
        return latitude;
    }

    public double getLongitude() {
        return longitude;
    }

    // Phương thức để cập nhật vị trí
    public void setLocation(double latitude, double longitude) {
        this.latitude = latitude;
        this.longitude = longitude;
    }
}
