package com.sagin.util;


import com.sagin.model.NodeInfo;
import com.sagin.network.implement.NodeGateway;
import com.sagin.network.implement.TCP_Service;
import com.sagin.repository.INodeRepository;
import com.sagin.repository.IUserRepository;
import com.sagin.repository.MongoNodeRepository;
import com.sagin.repository.MongoUserRepository;
import com.sagin.routing.IRoutingService;
import com.sagin.routing.RoutingService;
import com.sagin.service.INodeService;
import com.sagin.service.NodeService;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;

public class SimulationMain {

    private static final Logger logger = LoggerFactory.getLogger(SimulationMain.class);

    public static void main(String[] args) {
        // Khởi tạo repository & service
        INodeRepository nodeRepository = new MongoNodeRepository();
        IUserRepository userRepository = new MongoUserRepository();
        INodeService nodeService = new NodeService(nodeRepository);
        IRoutingService routingService = new RoutingService();
        TCP_Service tcpService = new TCP_Service(nodeRepository, nodeService, userRepository, routingService);

        // Khởi tạo NodeGateway
        NodeGateway nodeGateway = new NodeGateway(tcpService);

        // Lấy NodeInfo từ repository
        Optional<NodeInfo> nodeInfoOptional = nodeRepository.getNodeInfo("GS-01");

        if (nodeInfoOptional.isPresent()) {
            NodeInfo nodeInfo = nodeInfoOptional.get();
            int port = nodeInfo.getCommunication().port();

            // Start listener
            nodeGateway.startListening(nodeInfo, port);
            logger.info("Node Gateway started for node: {}", nodeInfo.getNodeId());

            // Thêm shutdown hook để stop NodeGateway an toàn khi JVM tắt
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                logger.info("Shutting down NodeGateway...");
                nodeGateway.stopListening();
            }));

            // Giữ main thread để listener tiếp tục chạy
            try {
                Thread.currentThread().join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

        } else {
            logger.error("Node with ID 'GW-Node-1' not found in the repository.");
        }
    }
}
