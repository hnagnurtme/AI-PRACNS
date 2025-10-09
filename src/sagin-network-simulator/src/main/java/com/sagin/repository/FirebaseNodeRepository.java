package com.sagin.repository;

import com.google.cloud.firestore.CollectionReference;
import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.google.cloud.firestore.QuerySnapshot;
import com.google.cloud.firestore.WriteResult;
import com.google.api.core.ApiFuture;

import com.sagin.configuration.FireStoreConfiguration;
import com.sagin.configuration.FirebaseConfiguration;
import com.sagin.model.NodeInfo;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.Map;
import java.util.List;
import java.util.HashMap;
import java.util.concurrent.ExecutionException;

/**
 * Triển khai INodeRepository sử dụng Google Firestore làm nguồn dữ liệu tập trung.
 * Lớp này chịu trách nhiệm mapping dữ liệu giữa Firestore và đối tượng NodeInfo.
 */
public class FirebaseNodeRepository implements INodeRepository {
    
    private static final Logger logger = LoggerFactory.getLogger(FirebaseNodeRepository.class);
    private final Firestore firestore;
    private final CollectionReference nodesCollection;

    public FirebaseNodeRepository() {
        this.firestore = FireStoreConfiguration.getFirestore();
        
        // Tham chiếu đến Collection chứa NodeInfo
        this.nodesCollection = firestore.collection(FirebaseConfiguration.NODES_COLLECTION_PATH);
        
        logger.info("FirebaseNodeRepository đã kết nối tới Collection: {}", 
                    FirebaseConfiguration.NODES_COLLECTION_PATH);
    }
    
    @Override
    public Map<String, NodeInfo> loadAllNodeConfigs() {
        Map<String, NodeInfo> configs = new HashMap<>();
        
        try {
            logger.info("Đang tải toàn bộ cấu hình Node từ Firestore...");
            
            // Thực hiện truy vấn bất đồng bộ (ApiFuture) và chờ kết quả
            ApiFuture<QuerySnapshot> future = nodesCollection.get();
            List<QueryDocumentSnapshot> documents = future.get().getDocuments();

            for (QueryDocumentSnapshot document : documents) {
                // Chuyển đổi document thành đối tượng NodeInfo
                // Firestore tự động sử dụng Jackson (thư viện bạn đã thêm) để mapping
                NodeInfo node = document.toObject(NodeInfo.class);
                
                // Gán Node ID từ Document ID (để đảm bảo tính chính xác)
                node.setNodeId(document.getId()); 
                configs.put(node.getNodeId(), node);
            }
            
            logger.info("Tải thành công {} cấu hình Node từ Database.", configs.size());
            
        } catch (InterruptedException | ExecutionException e) {
            logger.error("LỖI TRUY VẤN FIRESTORE: Không thể tải cấu hình Node.", e);
            return Collections.emptyMap(); // Trả về Map rỗng nếu có lỗi
        }
        
        return configs;
    }

    @Override
    public void updateNodeInfo(String nodeId, NodeInfo info) {
        // Chỉ cập nhật các trường thay đổi (ví dụ: position, powerLevel)
        
        // Giả sử NodeInfo.lastUpdated đã được cập nhật trước đó
        info.setLastUpdated(System.currentTimeMillis());

        try {
            // Lệnh set() hoặc update() để ghi dữ liệu trở lại Firestore
            ApiFuture<WriteResult> future = nodesCollection.document(nodeId).set(info);
            // Bạn có thể không cần chờ kết quả (future.get()) nếu muốn tốc độ nhanh hơn
            // Tuy nhiên, việc chờ giúp xác nhận dữ liệu đã ghi
            future.get(); 
            
            logger.debug("Cập nhật trạng thái thành công cho Node {}.", nodeId);
        } catch (InterruptedException | ExecutionException e) {
            logger.error("LỖI GHI DỮ LIỆU FIRESTORE cho Node {}: {}", nodeId, e.getMessage(), e);
        }
    }

    @Override
    public NodeInfo getNodeInfo(String nodeId) {
        try {
            logger.info("Đang tải thông tin Node {} từ Firestore...", nodeId);  
            ApiFuture<DocumentSnapshot> future = nodesCollection.document(nodeId).get();
            DocumentSnapshot document = future.get();    
            if (document.exists()) {
                NodeInfo node = document.toObject(NodeInfo.class);
                if (node != null) {
                    node.setNodeId(document.getId());
                    logger.info("Tải thành công thông tin Node {} từ Database.", nodeId);
                    return node;
                } else {
                    logger.warn("Dữ liệu Node {} không hợp lệ trong Database.", nodeId);
                    return null;
                }
            } else {
                logger.warn("Node {} không tồn tại trong Database.", nodeId);
                return null;
            }
        } catch (InterruptedException | ExecutionException e) {
            logger.error("LỖI TRUY VẤN FIRESTORE cho Node {}: {}", nodeId, e.getMessage(), e);
            return null;
        }
    }
}