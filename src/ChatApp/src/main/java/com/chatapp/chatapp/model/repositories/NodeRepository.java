package com.chatapp.chatapp.model.repositories;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

import com.chatapp.chatapp.model.entities.Node;
import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.QuerySnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;

public class NodeRepository {
    // Lấy toàn bộ documents trong collection nodes
    public static List<Node> getAllNodes(Firestore firestore) throws ExecutionException, InterruptedException {
        ApiFuture<QuerySnapshot> future = firestore.collection("nodes").get();
        List<QueryDocumentSnapshot> documents = future.get().getDocuments();
        List<Node> nodes = new ArrayList<>();
        for (QueryDocumentSnapshot doc : documents) {
            // Lấy thông tin position
            Double lat = doc.getDouble("position.latitude");
            Double lon = doc.getDouble("position.longitude");

            if (lat == null || lon == null) {
                System.out.println("Document " + doc.getId() + " is missing position data.");
                continue; 
            }

            String nodeId = doc.getString("nodeId");
            String nodeType = doc.getString("nodeType");
            if (nodeId == null || nodeType == null) {
                System.out.println("Document " + doc.getId() + " is missing nodeId or nodeType.");
                continue; 
            }

            nodes.add(new Node(nodeId, nodeType, lat, lon));
        }
        return nodes;
    }
}
