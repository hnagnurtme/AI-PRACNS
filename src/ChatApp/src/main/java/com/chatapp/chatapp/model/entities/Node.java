package com.chatapp.chatapp.model.entities;

public class Node {
    private String nodeId;
    private String nodeType;
    private double latitude;
    private double longitude;

    public Node(String nodeId, String nodeType, double latitude, double longitude) {
        this.nodeId = nodeId;
        this.nodeType = nodeType;
        this.latitude = latitude;
        this.longitude = longitude;
    }
    public String getNodeId() {
        return nodeId;
    }
    public String getNodeType() {
        return nodeType;
    }
    public double getLatitude() {
        return latitude;
    }
    public double getLongitude() {
        return longitude;
    }
}
