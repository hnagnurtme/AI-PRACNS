package com.chatapp.chatapp.application;

import java.util.List;

import com.chatapp.chatapp.gui.interfaces.NetworkTopologyService;

import javafx.scene.layout.Pane;

public class NetworkTopologyServiceImpl implements NetworkTopologyService {
    private final Pane topologyCanvas;
    
    public NetworkTopologyServiceImpl(Pane topologyCanvas) {
        this.topologyCanvas = topologyCanvas;
        setupTopologyCanvas();
    }
    
    private void setupTopologyCanvas() {
        if (topologyCanvas != null) {
            topologyCanvas.setPrefSize(800, 600);
            topologyCanvas.setStyle("-fx-background-color: #f0f0f0; -fx-border-color: #cccccc;");
        }
    }
    
    @Override
    public void updateTopology(List<NetworkNode> nodes, List<NetworkLink> links) {
        System.out.println("Updating topology with " + nodes.size() + " nodes and " + links.size() + " links");
    }
    
    @Override
    public void zoomIn() {
        zoomTopology(1.2);
    }
    
    @Override
    public void zoomOut() {
        zoomTopology(0.8);
    }
    
    @Override
    public void resetView() {
        if (topologyCanvas != null) {
            topologyCanvas.setScaleX(1.0);
            topologyCanvas.setScaleY(1.0);
            topologyCanvas.setTranslateX(0);
            topologyCanvas.setTranslateY(0);
        }
    }
    
    @Override
    public void hightlightPath(List<String> nodeIds) {
        System.out.println("Highlighting path: " + String.join(" -> ", nodeIds));
    }
    
    private void zoomTopology(double factor) {
        if (topologyCanvas != null) {
            topologyCanvas.setScaleX(topologyCanvas.getScaleX() * factor);
            topologyCanvas.setScaleY(topologyCanvas.getScaleY() * factor);
        }
    }
}