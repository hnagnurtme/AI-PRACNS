package com.sagsins.core.repository;

import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.Firestore;
import com.sagsins.core.model.NodeInfo;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.stream.StreamSupport;

@Repository
public class NodeRepository implements INodeRepository {

    private final Firestore firestore;

    public NodeRepository(Firestore firestore) {
        this.firestore = firestore;
    }

    @Override
    public List<NodeInfo> getAllNodes() {
        try {
            Iterable<DocumentReference> docs = firestore.collection("nodes").listDocuments();
            return StreamSupport.stream(docs.spliterator(), false) 
                    .map(docRef -> {
                        try {
                            return docRef.get().get().toObject(NodeInfo.class);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            e.printStackTrace();
                        } catch (ExecutionException e) {
                            e.printStackTrace();
                        }
                        return null;
                    })
                    .filter(node -> node != null)
                    .toList();
        } catch (Exception e) {
            e.printStackTrace();
            return List.of();
        }
    }

    @Override
    public NodeInfo getNodeById(String id) {
        try {
            return firestore.collection("nodes")
                    .document(id)
                    .get()
                    .get()
                    .toObject(NodeInfo.class);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public void saveNode(NodeInfo node) {
        try {
            firestore.collection("nodes")
                    .document(node.getNodeId())
                    .set(node)
                    .get(); // đợi hoàn tất
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void deleteNode(String nodeId) {
        try {
            firestore.collection("nodes")
                    .document(nodeId)
                    .delete()
                    .get(); // đợi hoàn tất
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
    }
}
