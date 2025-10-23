package com.sagin.util;

import com.sagin.model.NodeInfo;
import com.sagin.model.NodeType;
import com.sagin.network.implement.NodeGateway;
import com.sagin.network.implement.TCP_Service;
import com.sagin.repository.INodeRepository;
import com.sagin.repository.IUserRepository;
import com.sagin.repository.MongoNodeRepository;
import com.sagin.repository.MongoUserRepository;
import com.sagin.routing.DynamicRoutingService;
import com.sagin.service.INodeService;
import com.sagin.service.NodeService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.stream.Collectors;

public class SimulationMain {

    private static final Logger logger = LoggerFactory.getLogger(SimulationMain.class);

    public static void main(String[] args) {
        if (args.length == 0) {
            System.err.println("Cú pháp: java com.sagin.util.SimulationMain <NODE_TYPE>");
            System.err.println("Ví dụ: java com.sagin.util.SimulationMain LEO_SATELLITE");
            return;
        }

        NodeType typeToRun;
        try {
            typeToRun = NodeType.fromString(args[0]);
        } catch (IllegalArgumentException e) {
            System.err.println("Loại node không hợp lệ: " + args[0]);
            System.err.println("Các loại hợp lệ: GROUND_STATION, LEO_SATELLITE, MEO_SATELLITE, GEO_SATELLITE");
            return;
        }

        INodeRepository nodeRepository = new MongoNodeRepository();
        IUserRepository userRepository = new MongoUserRepository();
        INodeService nodeService = new NodeService(nodeRepository);
        DynamicRoutingService routingService = new DynamicRoutingService(nodeRepository, nodeService);

        logger.info("Khởi chạy mô phỏng cho loại node: {}", typeToRun);

        List<NodeInfo> nodesToRun = nodeRepository.getAllNodes().stream()
                .filter(n -> n.getNodeType() == typeToRun)
                .collect(Collectors.toList());

        if (nodesToRun.isEmpty()) {
            logger.warn("⚠️ Không tìm thấy node nào có type: {}", typeToRun);
            return;
        }

        logger.info("🔧 Số lượng node cần chạy: {}", nodesToRun.size());

        for (NodeInfo nodeInfo : nodesToRun) {
            TCP_Service tcpService = new TCP_Service(nodeRepository, nodeService, userRepository, routingService);
            NodeGateway nodeGateway = new NodeGateway(tcpService);

            new Thread(() -> {
                nodeGateway.startListening(nodeInfo, nodeInfo.getCommunication().port());
                logger.info("Node Gateway started for node: {}", nodeInfo.getNodeId());
                try {
                    Thread.currentThread().join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    logger.warn("Node {} listener interrupted", nodeInfo.getNodeId(), e);
                }
            }, "Thread-" + nodeInfo.getNodeId()).start();
        }

        // Shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info("Shutting down all NodeGateways...");
            routingService.shutdown();
        }));

        routingService.forceUpdateRoutingTables();
        logger.info("Initial routing tables updated");
    }
}
