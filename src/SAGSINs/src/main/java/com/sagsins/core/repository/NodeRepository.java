package com.sagsins.core.repository;

import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.sagsins.core.exception.NotFoundException;
import com.sagsins.core.model.NodeInfo;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ExecutionException;

@Repository
public class NodeRepository implements INodeRepository {

    private static final String COLLECTION_NAME = "nodes";
    private final Firestore firestore;

    public NodeRepository(Firestore firestore) {
        this.firestore = firestore;
    }

    @Override
    public NodeInfo save(NodeInfo node) {
        if (node.getNodeId() == null || node.getNodeId().isEmpty()) {
            String newId = firestore.collection(COLLECTION_NAME).document().getId();
            node.setNodeId(newId);
        }
        
        try {
            firestore.collection(COLLECTION_NAME)
                    .document(node.getNodeId())
                    .set(node)
                    .get();
            return node;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Save operation interrupted", e);
        } catch (ExecutionException e) {
            throw new RuntimeException("Error saving node to Firestore", e);
        }
    }

    @Override
    public Optional<NodeInfo> findById(String nodeId) {
        try {
            DocumentSnapshot document = firestore.collection(COLLECTION_NAME)
                    .document(nodeId)
                    .get()
                    .get();

            if (document.exists()) {
                return Optional.ofNullable(document.toObject(NodeInfo.class));
            } else {
                return Optional.empty();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new NotFoundException("FindById operation interrupted");
        } catch (ExecutionException e) {
            // Nếu Firestore ném ngoại lệ (ví dụ: lỗi kết nối), nên ném RuntimeException
            throw new NotFoundException("Error fetching node from Firestore: " + e.getMessage());
        }
    }
    
    @Override
    public List<NodeInfo> findAll() {
        try {
            List<QueryDocumentSnapshot> documents = firestore.collection(COLLECTION_NAME)
                    .get()
                    .get()
                    .getDocuments(); 

            return documents.stream() 
                    .map(document -> document.toObject(NodeInfo.class))
                    .toList();
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("FindAll operation interrupted", e);
        } catch (ExecutionException e) {
            throw new RuntimeException("Error fetching all nodes from Firestore", e);
        }
    }
    
    @Override
    public void deleteById(String nodeId) {
        try {
            firestore.collection(COLLECTION_NAME)
                    .document(nodeId)
                    .delete()
                    .get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new NotFoundException("Delete operation interrupted");
        } catch (ExecutionException e) {
            throw new NotFoundException("Error deleting node from Firestore: " + e.getMessage());
        }
    }
    
    @Override
    public boolean existsById(String nodeId) {
        try {
            DocumentSnapshot document = firestore.collection(COLLECTION_NAME)
                    .document(nodeId)
                    .get()
                    .get();
            return document.exists();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("ExistsById operation interrupted", e);
        } catch (ExecutionException e) {
            throw new RuntimeException("Error checking node existence", e);
        }
    }
}