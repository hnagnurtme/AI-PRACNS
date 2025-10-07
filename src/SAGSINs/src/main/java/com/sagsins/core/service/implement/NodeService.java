package com.sagsins.core.service.implement;

import com.sagsins.core.DTOs.CreateNodeRequest;
import com.sagsins.core.DTOs.NodeDTO;
import com.sagsins.core.DTOs.UpdateNodeRequest;
import com.sagsins.core.exception.DuplicateKeyException;
import com.sagsins.core.model.NodeInfo;
import com.sagsins.core.repository.INodeRepository;
import com.sagsins.core.service.INodeService;

import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Triển khai interface INodeService, chứa logic nghiệp vụ và giao tiếp với Repository.
 */
@Service
public class NodeService implements INodeService {

    private final INodeRepository nodeRepository;

    public NodeService(INodeRepository nodeRepository) {
        this.nodeRepository = nodeRepository;
    }
// ----------------------------------------------------------------------
    // --- CREATE ---
    @Override
    public NodeDTO createNode(CreateNodeRequest request) {

        NodeInfo newNode = new NodeInfo();
        
        boolean exists = nodeRepository.existsById(request.getNodeId());
        if (exists) {
            throw new DuplicateKeyException("Node with ID " + request.getNodeId() + " already exists.");
        }
        newNode.setNodeId(request.getNodeId());
        newNode.setNodeType(request.getNodeType());
        newNode.setOperational(request.isOperational());

        if (request.getPosition() != null) {
            newNode.setPosition(request.getPosition());
        }
        if (request.getOrbit() != null) {
                newNode.setOrbit(request.getOrbit());
        }
        if (request.getVelocity() != null) {
            newNode.setVelocity(request.getVelocity());
        }
        
        // Cài đặt Metrics ban đầu
        newNode.setCurrentBandwidth(request.getCurrentBandwidth());
        newNode.setAvgLatencyMs(request.getAvgLatencyMs());
        newNode.setPacketLossRate(request.getPacketLossRate());
        newNode.setPacketBufferLoad(0); 
        newNode.setCurrentThroughput(0); 
        newNode.setResourceUtilization(0); 
        newNode.setPowerLevel(100.0); 
        
        newNode.setLastUpdated(System.currentTimeMillis());

        // 2. Lưu Entity vào Repository (DB)
        NodeInfo savedNode = nodeRepository.save(newNode);
        
        // 3. Chuyển đổi Entity đã lưu sang DTO để trả về
        return convertToDTO(savedNode); // Sử dụng hàm convertToDTO đã định nghĩa
    }

    @Override
    public List<NodeDTO> getAllNodes() {
        // Lấy List<NodeInfo> từ Repository, sau đó chuyển đổi sang List<NodeDTO>
        return nodeRepository.findAll().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    @Override
    public Optional<NodeDTO> getNodeById(String nodeId) {
        return nodeRepository.findById(nodeId)
                .map(this::convertToDTO);
    }

    @Override
    public Optional<NodeDTO> updateNode(String nodeId, UpdateNodeRequest request) {
        return nodeRepository.findById(nodeId).map(existingNode -> {
            
            // Cập nhật các trường cấu hình
            if (request.getNodeType() != null) {
                existingNode.setNodeType(request.getNodeType());
            }
            if (request.getIsOperational() != null) {
                existingNode.setOperational(request.getIsOperational());
            }

            // Cập nhật Vị trí/Cơ học
            if (request.getPosition() != null) {
                existingNode.setPosition(request.getPosition());
            }
            if (request.getOrbit() != null) {
                existingNode.setOrbit(request.getOrbit());
            }
            if (request.getVelocity() != null) {
                existingNode.setVelocity(request.getVelocity());
            }

            // Cập nhật các Metrics QoS (chỉ cập nhật nếu Request cung cấp)
            if (request.getCurrentBandwidth() != null) { 
                existingNode.setCurrentBandwidth(request.getCurrentBandwidth());
            }
            if (request.getAvgLatencyMs() != null) {
                existingNode.setAvgLatencyMs(request.getAvgLatencyMs());
            }
            if (request.getPacketLossRate() != null) {
                existingNode.setPacketLossRate(request.getPacketLossRate());
            }
            if (request.getPacketBufferLoad() != null) {
                existingNode.setPacketBufferLoad(request.getPacketBufferLoad());
            }
            if (request.getCurrentThroughput() != null) {
                existingNode.setCurrentThroughput(request.getCurrentThroughput());
            }
            if (request.getResourceUtilization() != null) {
                existingNode.setResourceUtilization(request.getResourceUtilization());
            }
            if (request.getPowerLevel() != null) {
                existingNode.setPowerLevel(request.getPowerLevel());
            }
            
            existingNode.setLastUpdated(System.currentTimeMillis());

            NodeInfo updatedNode = nodeRepository.save(existingNode);
            return convertToDTO(updatedNode);

        });
    }
    @Override
    public boolean deleteNode(String nodeId) {
        if (nodeRepository.existsById(nodeId)) {
            nodeRepository.deleteById(nodeId);
            return true;
        }
        return false;
    }

    private NodeDTO convertToDTO(NodeInfo nodeInfo) {
        if (nodeInfo == null) return null;

        NodeDTO dto = new NodeDTO();
        
        dto.setNodeId(nodeInfo.getNodeId());
        dto.setNodeType(nodeInfo.getNodeType());
        dto.setPosition(nodeInfo.getPosition());
        dto.setOrbit(nodeInfo.getOrbit());
        dto.setVelocity(nodeInfo.getVelocity());

        dto.setOperational(nodeInfo.isOperational());
        // Sử dụng phương thức tính toán từ NodeInfo
        dto.setIsHealthy(nodeInfo.isHealthy()); 

        dto.setCurrentBandwidth(nodeInfo.getCurrentBandwidth());
        dto.setAvgLatencyMs(nodeInfo.getAvgLatencyMs());
        dto.setPacketLossRate(nodeInfo.getPacketLossRate());
        dto.setCurrentThroughput(nodeInfo.getCurrentThroughput());
        dto.setResourceUtilization(nodeInfo.getResourceUtilization());
        dto.setPowerLevel(nodeInfo.getPowerLevel());
        
        dto.setLastUpdated(nodeInfo.getLastUpdated());

        return dto;
    }
}