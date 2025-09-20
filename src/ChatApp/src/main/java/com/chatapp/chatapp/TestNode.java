package com.chatapp.chatapp;

import java.util.concurrent.ExecutionException;

import com.chatapp.auth.model.entities.User;
import com.chatapp.chatapp.config.FirebaseConfig;
import com.chatapp.chatapp.model.entities.Node;
import com.chatapp.chatapp.model.services.NodeService;
import com.google.cloud.firestore.Firestore;

public class TestNode {
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        // 1. Khởi tạo Firestore
        Firestore db = FirebaseConfig.getFirestore();
        // 3. User mock: ví dụ đang ở Hà Nội
        double userLat = 10.8220;
        double userLon = 106.6257;

        User user = new User("luongvanvo29@gmail.com", null, userLat, userLon);
        // 4. Tìm node gần nhất
        NodeService nodeService = new NodeService(db);
        Node nearest = nodeService.findNearestNode(user);

        System.out.println("Nearest node: " + nearest.getNodeId() +
                           " type: " + nearest.getNodeType());
    }
}
