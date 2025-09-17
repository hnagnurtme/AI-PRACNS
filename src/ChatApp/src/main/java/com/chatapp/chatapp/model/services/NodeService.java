package com.chatapp.chatapp.model.services;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

import com.chatapp.auth.model.entities.User;
import com.chatapp.chatapp.model.entities.Node;
import com.chatapp.chatapp.model.interfaces.INodeService;
import com.chatapp.chatapp.model.repositories.NodeRepository;
import com.google.cloud.firestore.Firestore;

public class NodeService implements INodeService {

    private final Firestore firestore;

    public NodeService(Firestore firestore) {
        this.firestore = firestore;
    }
    @Override
    public Node findNearestNode(User user) throws ExecutionException, InterruptedException {
        // Lấy tất cả nodes từ Firestore
        List<Node> nodes = NodeRepository.getAllNodes(firestore);
        if (nodes.isEmpty()) {
            System.out.println("No nodes found.");
            return null; // Hoặc ném ngoại lệ nếu không có node nào
        }

        // Duyệt và tìm node gần nhất
        Node nearest = null;
        double minDistance = Double.MAX_VALUE;

        for (Node node : nodes) {
            double distance = haversine(user.getLatitude(), user.getLongitude(), node.getLatitude(), node.getLongitude());

            if (distance < minDistance) {
                minDistance = distance;
                nearest = node;
            }
        }
        return nearest;
    }

    @Override
    public double haversine(double lat1, double lon1, double lat2, double lon2) {
        final int R = 6371; // Radius of the earth in km
        double dLat = Math.toRadians(lat2 - lat1);
        double dLon = Math.toRadians(lon2 - lon1);

        double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) + Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        double distance = R * c; // Distance in km
        return distance;
    }
}
