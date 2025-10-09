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
        // Query cả hai collections song song
        ApiFuture<QuerySnapshot> groundStationsFuture = firestore.collection("ground_stations").get();
        ApiFuture<QuerySnapshot> seaStationsFuture = firestore.collection("sea_stations").get();

        List<Node> nodes = new ArrayList<>();

        // Chờ cả hai queries hoàn thành
        List<QueryDocumentSnapshot> groundStationsDocuments = groundStationsFuture.get().getDocuments();
        List<QueryDocumentSnapshot> seaStationsDocuments = seaStationsFuture.get().getDocuments();

        // Process all documents
        for (QueryDocumentSnapshot doc : groundStationsDocuments) {
            Node node = extractNodeFromDocument(doc);
            if (node != null)
                nodes.add(node);
        }

        for (QueryDocumentSnapshot doc : seaStationsDocuments) {
            Node node = extractNodeFromDocument(doc);
            if (node != null)
                nodes.add(node);
        }

        return nodes;
    }

    private static Node extractNodeFromDocument(QueryDocumentSnapshot doc) {
        // Lấy thông tin position
        Double lat = doc.getDouble("position.latitude");
        Double lon = doc.getDouble("position.longitude");

        if (lat == null || lon == null) {
            System.out.println("Document " + doc.getId() + " is missing position data.");
            return null;
        }

        String nodeId = doc.getString("nodeId");
        String nodeType = doc.getString("nodeType");
        if (nodeId == null || nodeType == null) {
            System.out.println("Document " + doc.getId() + " is missing nodeId or nodeType.");
            return null;
        }

        return new Node(nodeId, nodeType, lat, lon);
    }
}
