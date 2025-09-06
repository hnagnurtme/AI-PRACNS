package com.chatapp.chatapp.model.interfaces;

import java.util.List;

public interface NetworkTopologyService {
    void updateTopology(List<NetworkNode> nodes, List<NetworkLink> links);
    void zoomIn();
    void zoomOut();
    void resetView();
    void hightlightPath(List<String> nodeIds);

    // Inner classes form data models
    record NetworkNode(String id, String type, double x, double y, String status) {}
    record NetworkLink(String source, String target, String status, double bandwidth) {}
}
