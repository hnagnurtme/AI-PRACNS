package com.sagin.satellite.service.implement;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sagin.satellite.model.Packet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class TcpSender {
    private static final Logger logger = LoggerFactory.getLogger(TcpSender.class);
    private final ObjectMapper mapper = new ObjectMapper();
    private final int sendTimeoutMs = 2000;
    private final Map<String, String> nodeAddressCache = new ConcurrentHashMap<>();

    public void send(Packet packet) throws Exception {
        
    }
}
