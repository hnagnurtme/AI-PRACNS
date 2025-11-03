package com.sagin.routing;

import com.sagin.DTOs.RoutingRequest;
import com.sagin.DTOs.RoutingResponse;
import com.sagin.model.Packet;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.net.Socket;

public class RLRoutingService {

    private final String rlHost;
    private final int rlPort;
    private final ObjectMapper mapper = new ObjectMapper();

    public RLRoutingService(String rlHost, int rlPort) {
        this.rlHost = rlHost;
        this.rlPort = rlPort;
    }

    public RouteInfo getNextHop(RoutingRequest request) {
        try (Socket socket = new Socket(rlHost, rlPort)) {
            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));

            Packet packetRequest = new Packet();
            packetRequest.setPacketId(request.getPacketId());
            packetRequest.setCurrentHoldingNodeId(request.getCurrentHoldingNodeId());
            packetRequest.setStationDest(request.getStationDest());
            packetRequest.setServiceQoS(request.getServiceQoS());
            packetRequest.setTTL(request.getTtl());
            packetRequest.setUseRL(true);
            packetRequest.setMaxAcceptableLatencyMs(request.getAccumulatedDelayMs());


            System.out.println("Gửi yêu cầu routing đến RL service: " + rlHost + ":" + rlPort);
            System.out.println(packetRequest);

            // Gửi JSON
            String requestJson = mapper.writeValueAsString(packetRequest);
            out.println(requestJson);

            // Nhận JSON response
            String responseJson = in.readLine();

            System.out.println("Đã nhận phản hồi từ RL service:");
            System.out.println(responseJson);
            RoutingResponse routingResponse = mapper.readValue(responseJson, RoutingResponse.class);

            System.out.println("RoutingResponse: " + routingResponse.toString());

            RouteInfo routeInfo = RouteInfo.builder()
                    .nextHopNodeId(routingResponse.getNextHopNodeId())
                    .pathNodeIds(routingResponse.getPath())
                    .build();
            
            return routeInfo;

        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
