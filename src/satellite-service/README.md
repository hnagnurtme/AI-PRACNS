# Satellite Service - SAGIN Network Node

## Tổng quan
Satellite Service là một node trong mạng SAGIN (Satellite-Air-Ground Integrated Network), có nhiệm vụ xử lý và chuyển tiếp gói tin giữa các node vệ tinh, máy bay, và trạm mặt đất. Service này được thiết kế để xử lý QoS động dựa trên điều kiện thời tiết và cung cấp khả năng buffer packet khi đường truyền bị quá tải.

## Cấu trúc thư mục (hoàn chỉnh)

```
satellite-service/
├─ pom.xml
├─ resources/
│   └─ application.properties
├─ src/main/java/
│   ├─ domain/
│   │   ├─ Packet.java
│   │   ├─ RoutingTable.java
│   │   ├─ LinkStatus.java
│   │   ├─ NodeInfo.java
│   │   └─ Coordinates.java
│   ├─ application/
│   │   ├─ Forwarder.java
│   │   ├─ QoSModule.java
│   │   ├─ BufferManager.java
│   │   └─ StateManager.java
│   ├─ infrastructure/
│   │   ├─ FirebaseReporter.java
│   │   ├─ TcpServer.java
│   │   ├─ TcpSender.java
│   │   └─ ManagerClient.java
│   └─ interface/
│       └─ SatelliteApp.java
└─ README.md
```

## Maven Dependencies (pom.xml)

```xml
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.sagin</groupId>
  <artifactId>satellite-service</artifactId>
  <version>0.1.0</version>
  <properties>
    <maven.compiler.source>17</maven.compiler.source>
    <maven.compiler.target>17</maven.compiler.target>
  </properties>
  <dependencies>
    <!-- JSON -->
    <dependency>
      <groupId>com.fasterxml.jackson.core</groupId>
      <artifactId>jackson-databind</artifactId>
      <version>2.15.2</version>
    </dependency>

    <!-- Firebase Admin SDK -->
    <dependency>
      <groupId>com.google.firebase</groupId>
      <artifactId>firebase-admin</artifactId>
      <version>9.1.1</version>
    </dependency>

    <!-- Google Auth -->
    <dependency>
      <groupId>com.google.auth</groupId>
      <artifactId>google-auth-library-credentials</artifactId>
      <version>1.18.0</version>
    </dependency>

    <!-- SLF4J + simple logger -->
    <dependency>
      <groupId>org.slf4j</groupId>
      <artifactId>slf4j-api</artifactId>
      <version>2.0.7</version>
    </dependency>
    <dependency>
      <groupId>org.slf4j</groupId>
      <artifactId>slf4j-simple</artifactId>
      <version>2.0.7</version>
    </dependency>

    <!-- HTTP client -->
    <dependency>
      <groupId>com.squareup.okhttp3</groupId>
      <artifactId>okhttp</artifactId>
      <version>4.11.0</version>
    </dependency>

    <!-- JUnit -->
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>5.10.0</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
</project>
```

## Configuration (resources/application.properties)

```properties
# Node configuration
node.id=sat1
node.ip=10.0.0.2
node.port=9090

# Manager API
manager.api.url=http://localhost:8081

# Firebase Admin SDK
firebase.serviceAccountPath=/path/to/firebase-service-account.json
firebase.databaseUrl=https://your-project-id-default-rtdb.firebaseio.com

# Buffer & state config
buffer.maxSize=500
state.report.interval.ms=5000
```

## Domain Layer - Models & Routing

### Packet.java - Core packet model với metadata
```java
package domain;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.UUID;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class Packet {
    private String id;
    private String src;
    private String dst;
    private int payloadSize;
    private int ttl;
    private String currentNode;
    private String nextHop;
    private long createdAt;
    private long lastUpdated;
    private double delayMs;
    private double lossRate;
    private boolean dropped;

    public Packet() { /* Jackson */ }

    public Packet(String src, String dst, int payloadSize) {
        this.id = UUID.randomUUID().toString();
        this.src = src;
        this.dst = dst;
        this.payloadSize = payloadSize;
        this.ttl = 10;
        this.createdAt = System.currentTimeMillis();
        this.lastUpdated = this.createdAt;
        this.dropped = false;
        this.delayMs = 0.0;
        this.lossRate = 0.0;
    }

    // Getters & Setters (generate via IDE)
    public String getId() { return id; }
    public String getSrc() { return src; }
    public void setSrc(String src) { this.src = src; }
    public String getDst() { return dst; }
    public void setDst(String dst) { this.dst = dst; }
    public int getPayloadSize() { return payloadSize; }
    public void setPayloadSize(int payloadSize) { this.payloadSize = payloadSize; }
    public int getTtl() { return ttl; }
    public void setTtl(int ttl) { this.ttl = ttl; }
    public String getCurrentNode() { return currentNode; }
    public void setCurrentNode(String currentNode) { this.currentNode = currentNode; }
    public String getNextHop() { return nextHop; }
    public void setNextHop(String nextHop) { this.nextHop = nextHop; }
    public long getCreatedAt() { return createdAt; }
    public long getLastUpdated() { return lastUpdated; }
    public void setLastUpdated(long lastUpdated) { this.lastUpdated = lastUpdated; }
    public double getDelayMs() { return delayMs; }
    public void setDelayMs(double delayMs) { this.delayMs = delayMs; }
    public double getLossRate() { return lossRate; }
    public void setLossRate(double lossRate) { this.lossRate = lossRate; }
    public boolean isDropped() { return dropped; }
    public void setDropped(boolean dropped) { this.dropped = dropped; }
}
```

### RoutingTable.java - Thread-safe routing table
```java
package domain;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class RoutingTable {
    private final Map<String, String> routes = new ConcurrentHashMap<>();

    public void addRoute(String dst, String nextHop) {
        if (dst == null || nextHop == null) return;
        routes.put(dst, nextHop);
    }

    public String getNextHop(String dst) {
        return routes.get(dst);
    }

    public void removeRoute(String dst) {
        routes.remove(dst);
    }
}
```

### LinkStatus.java - Link metrics
```java
package domain;

public class LinkStatus {
    private String srcNode;
    private String dstNode;
    private double latencyMs;
    private double lossRate;
    private double capacityMbps;
    private boolean active;
    private double currentLoad;
    private long lastUpdated;

    public LinkStatus() { }

    // Getters & Setters
    public String getSrcNode() { return srcNode; }
    public void setSrcNode(String srcNode) { this.srcNode = srcNode; }
    public String getDstNode() { return dstNode; }
    public void setDstNode(String dstNode) { this.dstNode = dstNode; }
    public double getLatencyMs() { return latencyMs; }
    public void setLatencyMs(double latencyMs) { this.latencyMs = latencyMs; }
    public double getLossRate() { return lossRate; }
    public void setLossRate(double lossRate) { this.lossRate = lossRate; }
    public double getCapacityMbps() { return capacityMbps; }
    public void setCapacityMbps(double capacityMbps) { this.capacityMbps = capacityMbps; }
    public boolean isActive() { return active; }
    public void setActive(boolean active) { this.active = active; }
    public double getCurrentLoad() { return currentLoad; }
    public void setCurrentLoad(double currentLoad) { this.currentLoad = currentLoad; }
    public long getLastUpdated() { return lastUpdated; }
    public void setLastUpdated(long lastUpdated) { this.lastUpdated = lastUpdated; }
}
```

### NodeInfo.java & Coordinates.java - Node information
```java
package domain;

public class NodeInfo {
    private String id;
    private String type = "satellite";
    private String ip;
    private Coordinates coords;
    private boolean active;
    private double bandwidth;
    private double load;
    private long lastUpdated;

    public NodeInfo() { }

    // Getters & Setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public String getIp() { return ip; }
    public void setIp(String ip) { this.ip = ip; }
    public Coordinates getCoords() { return coords; }
    public void setCoords(Coordinates coords) { this.coords = coords; }
    public boolean isActive() { return active; }
    public void setActive(boolean active) { this.active = active; }
    public double getBandwidth() { return bandwidth; }
    public void setBandwidth(double bandwidth) { this.bandwidth = bandwidth; }
    public double getLoad() { return load; }
    public void setLoad(double load) { this.load = load; }
    public long getLastUpdated() { return lastUpdated; }
    public void setLastUpdated(long lastUpdated) { this.lastUpdated = lastUpdated; }
}

public class Coordinates {
    private double x, y, z;
    
    public Coordinates() {}
    public Coordinates(double x, double y, double z) { 
        this.x = x; this.y = y; this.z = z; 
    }
    
    public double getX() { return x; }
    public void setX(double x) { this.x = x; }
    public double getY() { return y; }
    public void setY(double y) { this.y = y; }
    public double getZ() { return z; }
    public void setZ(double z) { this.z = z; }
}
```

## Application Layer - Business Logic

### QoSModule.java - Weather-based QoS
```java
package application;

import domain.Packet;
import java.util.concurrent.atomic.AtomicReference;
import java.util.Random;

public class QoSModule {
    private final AtomicReference<Double> latencyFactor = new AtomicReference<>(1.0);
    private final AtomicReference<Double> lossFactor = new AtomicReference<>(0.0);
    private final Random random = new Random();

    public Packet apply(Packet packet, double baseLatencyMs, double baseLoss) {
        double lf = latencyFactor.get();
        double lfLoss = lossFactor.get();
        double delay = baseLatencyMs * lf;
        packet.setDelayMs(packet.getDelayMs() + delay);

        double combinedLoss = baseLoss + lfLoss;
        packet.setLossRate(packet.getLossRate() + combinedLoss);

        if (random.nextDouble() < combinedLoss) {
            packet.setDropped(true);
        }
        packet.setLastUpdated(System.currentTimeMillis());
        return packet;
    }

    public void updateWeatherFactors(double latencyFactor, double lossFactor) {
        this.latencyFactor.set(latencyFactor);
        this.lossFactor.set(lossFactor);
    }

    public double getLatencyFactor() { return latencyFactor.get(); }
    public double getLossFactor() { return lossFactor.get(); }
}
```

### BufferManager.java - FIFO packet buffering
```java
package application;

import domain.Packet;
import infrastructure.TcpSender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Queue;
import java.util.concurrent.*;

public class BufferManager {
    private static final Logger logger = LoggerFactory.getLogger(BufferManager.class);

    private final Queue<Packet> queue = new ConcurrentLinkedQueue<>();
    private final int maxSize;
    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
    private final TcpSender sender;

    public BufferManager(int maxSize, TcpSender sender) {
        this.maxSize = maxSize;
        this.sender = sender;
        scheduler.scheduleAtFixedRate(this::flush, 200, 200, TimeUnit.MILLISECONDS);
    }

    public boolean shouldBuffer() {
        return queue.size() >= maxSize;
    }

    public boolean enqueue(Packet packet) {
        if (queue.size() >= maxSize) {
            logger.warn("Buffer full — dropping packet {}", packet.getId());
            return false;
        }
        queue.offer(packet);
        return true;
    }

    private void flush() {
        try {
            Packet p;
            while ((p = queue.poll()) != null) {
                try {
                    sender.send(p);
                } catch (Exception ex) {
                    logger.error("Failed to send buffered packet {}, requeueing", p.getId(), ex);
                    queue.offer(p);
                    break;
                }
            }
        } catch (Exception e) {
            logger.error("Buffer flush error", e);
        }
    }

    public void shutdown() {
        scheduler.shutdownNow();
    }
}
```

### Forwarder.java - Core packet processing pipeline
```java
package application;

import domain.Packet;
import domain.RoutingTable;
import infrastructure.TcpSender;
import infrastructure.FirebaseReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Forwarder {
    private static final Logger logger = LoggerFactory.getLogger(Forwarder.class);

    private final RoutingTable routingTable;
    private final QoSModule qos;
    private final BufferManager buffer;
    private final TcpSender sender;
    private final FirebaseReporter reporter;
    private final String nodeId;

    public Forwarder(String nodeId, RoutingTable routingTable, QoSModule qos, 
                     BufferManager buffer, TcpSender sender, FirebaseReporter reporter) {
        this.nodeId = nodeId;
        this.routingTable = routingTable;
        this.qos = qos;
        this.buffer = buffer;
        this.sender = sender;
        this.reporter = reporter;
    }

    public void process(Packet packet) {
        try {
            if (packet == null) return;
            
            // TTL check
            if (packet.isDropped() || packet.getTtl() <= 0) {
                packet.setDropped(true);
                reporter.logDrop(packet);
                return;
            }

            // Routing lookup
            String nextHop = routingTable.getNextHop(packet.getDst());
            if (nextHop == null) {
                logger.warn("No route for dst={} — dropping packet {}", packet.getDst(), packet.getId());
                packet.setDropped(true);
                reporter.logDrop(packet);
                return;
            }

            packet.setNextHop(nextHop);
            packet.setCurrentNode(nodeId);

            // QoS application
            double baseLatencyMs = 20.0;
            double baseLoss = 0.001;
            packet = qos.apply(packet, baseLatencyMs, baseLoss);

            if (packet.isDropped()) {
                reporter.logDrop(packet);
                return;
            }

            // Buffer or send decision
            boolean bufferDecision = buffer.shouldBuffer();
            if (bufferDecision) {
                boolean enqueued = buffer.enqueue(packet);
                if (!enqueued) {
                    packet.setDropped(true);
                    reporter.logDrop(packet);
                } else {
                    reporter.logPacketBuffered(packet);
                }
            } else {
                sender.send(packet);
                reporter.logPacket(packet);
            }
        } catch (Exception ex) {
            logger.error("Error processing packet {}", packet == null ? "null" : packet.getId(), ex);
        }
    }
}
```

### StateManager.java - Periodic state reporting
```java
package application;

import domain.NodeInfo;
import domain.LinkStatus;
import infrastructure.FirebaseReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.*;

public class StateManager {
    private static final Logger logger = LoggerFactory.getLogger(StateManager.class);

    private final FirebaseReporter reporter;
    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
    private final long intervalMs;

    public StateManager(FirebaseReporter reporter, long intervalMs) {
        this.reporter = reporter;
        this.intervalMs = intervalMs;
    }

    public void startPeriodic(NodeInfo node, LinkStatus[] links) {
        scheduler.scheduleAtFixedRate(() -> {
            try {
                node.setLastUpdated(System.currentTimeMillis());
                reporter.updateNode(node);
                for (LinkStatus link : links) {
                    link.setLastUpdated(System.currentTimeMillis());
                    reporter.updateLink(link);
                }
            } catch (Exception e) {
                logger.error("State report failed", e);
            }
        }, 0, intervalMs, TimeUnit.MILLISECONDS);
    }

    public void stop() {
        scheduler.shutdownNow();
    }
}
```

## Infrastructure Layer - External Services

### FirebaseReporter.java - Firebase Realtime Database integration
```java
package infrastructure;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.api.core.ApiFuture;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import com.google.firebase.database.*;
import domain.LinkStatus;
import domain.NodeInfo;
import domain.Packet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.util.HashMap;
import java.util.Map;

public class FirebaseReporter {
    private static final Logger logger = LoggerFactory.getLogger(FirebaseReporter.class);
    private final DatabaseReference rootRef;
    private final ObjectMapper mapper = new ObjectMapper();

    public FirebaseReporter(String serviceAccountPath, String databaseUrl) throws Exception {
        FileInputStream serviceAccount = new FileInputStream(serviceAccountPath);
        FirebaseOptions options = FirebaseOptions.builder()
                .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                .setDatabaseUrl(databaseUrl)
                .build();

        if (FirebaseApp.getApps().isEmpty()) {
            FirebaseApp.initializeApp(options);
        }
        rootRef = FirebaseDatabase.getInstance().getReference();
    }

    public void logPacket(Packet packet) {
        try {
            Map<String, Object> payload = createPacketPayload(packet, "in-flight");
            DatabaseReference logsRef = rootRef.child("logs").push();
            ApiFuture<Void> future = logsRef.setValueAsync(payload);
            future.addListener(() -> logger.debug("Logged packet {}", packet.getId()), Runnable::run);
        } catch (Exception e) {
            logger.error("logPacket failed", e);
        }
    }

    public void logDrop(Packet packet) {
        try {
            Map<String, Object> payload = createPacketPayload(packet, "dropped");
            rootRef.child("logs").push().setValueAsync(payload);
        } catch (Exception e) {
            logger.error("logDrop failed", e);
        }
    }

    public void logPacketBuffered(Packet packet) {
        try {
            Map<String, Object> payload = createPacketPayload(packet, "buffered");
            rootRef.child("logs").push().setValueAsync(payload);
        } catch (Exception e) {
            logger.error("logPacketBuffered failed", e);
        }
    }

    private Map<String, Object> createPacketPayload(Packet packet, String status) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("packetId", packet.getId());
        payload.put("src", packet.getSrc());
        payload.put("dst", packet.getDst());
        payload.put("path", new String[]{packet.getSrc(), packet.getCurrentNode(), 
                                       packet.getNextHop(), packet.getDst()});
        payload.put("timestamp", System.currentTimeMillis());
        payload.put("status", status);
        payload.put("delayMs", packet.getDelayMs());
        payload.put("lossRate", packet.getLossRate());
        return payload;
    }

    public void updateNode(NodeInfo node) {
        try {
            Map<String, Object> nodeMap = mapper.convertValue(node, Map.class);
            Map<String, Object> payload = new HashMap<>(nodeMap);
            payload.put("status", node.isActive() ? "active" : "inactive");
            rootRef.child("nodes").child(node.getId()).setValueAsync(payload)
                    .addListener(() -> logger.debug("Updated node {}", node.getId()), Runnable::run);
        } catch (Exception e) {
            logger.error("updateNode failed", e);
        }
    }

    public void updateLink(LinkStatus link) {
        try {
            Map<String, Object> linkMap = mapper.convertValue(link, Map.class);
            String linkId = link.getSrcNode() + "_" + link.getDstNode();
            Map<String, Object> payload = new HashMap<>(linkMap);
            payload.put("status", link.isActive() ? "up" : "down");
            rootRef.child("links").child(linkId).setValueAsync(payload)
                    .addListener(() -> logger.debug("Updated link {}", linkId), Runnable::run);
        } catch (Exception e) {
            logger.error("updateLink failed", e);
        }
    }

    public void shutdown() {
        try {
            Thread.sleep(200); // Allow pending writes
        } catch (InterruptedException ignored) {}
    }
}
```

### TcpSender.java - TCP packet transmission
```java
package infrastructure;

import com.fasterxml.jackson.databind.ObjectMapper;
import domain.Packet;
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
    private final ManagerClient managerClient;
    private final int sendTimeoutMs = 2000;
    private final Map<String, String> nodeAddressCache = new ConcurrentHashMap<>();

    public TcpSender(ManagerClient managerClient) {
        this.managerClient = managerClient;
    }

    public void send(Packet packet) throws Exception {
        String nextHop = packet.getNextHop();
        if (nextHop == null) throw new IllegalArgumentException("nextHop missing");

        String address = nodeAddressCache.computeIfAbsent(nextHop, 
            k -> managerClient.getNodeAddress(k));
        if (address == null) {
            logger.warn("Unknown nextHop address for {}", nextHop);
            throw new IllegalStateException("Missing node address");
        }
        
        final String[] parts = address.split(":");
        final String ip = parts[0];
        final int port = Integer.parseInt(parts[1]);

        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(ip, port), sendTimeoutMs);
            OutputStream out = socket.getOutputStream();
            String json = mapper.writeValueAsString(packet);
            out.write(json.getBytes(StandardCharsets.UTF_8));
            out.write('\n');
            out.flush();
            logger.debug("Sent packet {} to {}:{}", packet.getId(), ip, port);
        } catch (Exception ex) {
            logger.error("Failed to send packet {} to {}: {}", packet.getId(), address, ex.getMessage());
            throw ex;
        }
    }
}
```

### TcpServer.java - Multi-threaded TCP server
```java
package infrastructure;

import application.Forwarder;
import com.fasterxml.jackson.databind.ObjectMapper;
import domain.Packet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.*;

public class TcpServer {
    private static final Logger logger = LoggerFactory.getLogger(TcpServer.class);
    private final Forwarder forwarder;
    private final ObjectMapper mapper = new ObjectMapper();
    private final ExecutorService workerPool = Executors.newFixedThreadPool(8);
    private ServerSocket serverSocket;

    public TcpServer(Forwarder forwarder) {
        this.forwarder = forwarder;
    }

    public void start(int port) throws IOException {
        serverSocket = new ServerSocket(port);
        logger.info("TcpServer listening on {}", port);
        Thread acceptThread = new Thread(() -> {
            while (!serverSocket.isClosed()) {
                try {
                    Socket client = serverSocket.accept();
                    workerPool.submit(() -> handle(client));
                } catch (IOException e) {
                    if (serverSocket.isClosed()) break;
                    logger.error("Accept failed", e);
                }
            }
        }, "TcpServer-Accept");
        acceptThread.start();
    }

    private void handle(Socket client) {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(client.getInputStream()))) {
            String line;
            while ((line = br.readLine()) != null) {
                try {
                    Packet packet = mapper.readValue(line, Packet.class);
                    forwarder.process(packet);
                } catch (Exception ex) {
                    logger.error("Invalid packet payload", ex);
                }
            }
        } catch (Exception e) {
            logger.debug("Connection closed from {}", client.getRemoteSocketAddress());
        }
    }

    public void stop() throws IOException {
        serverSocket.close();
        workerPool.shutdownNow();
    }
}
```

### ManagerClient.java - Manager API integration
```java
package infrastructure;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import okhttp3.*;

import java.util.concurrent.TimeUnit;

public class ManagerClient {
    private static final Logger logger = LoggerFactory.getLogger(ManagerClient.class);
    private final OkHttpClient http = new OkHttpClient.Builder()
            .connectTimeout(2, TimeUnit.SECONDS)
            .readTimeout(2, TimeUnit.SECONDS)
            .build();
    private final ObjectMapper mapper = new ObjectMapper();
    private final String managerApiUrl;

    public ManagerClient(String managerApiUrl) {
        this.managerApiUrl = managerApiUrl;
    }

    public String getNodeAddress(String nodeId) {
        try {
            Request req = new Request.Builder().url(managerApiUrl + "/api/nodes").build();
            try (Response res = http.newCall(req).execute()) {
                if (!res.isSuccessful()) return null;
                JsonNode arr = mapper.readTree(res.body().string());
                for (JsonNode n : arr) {
                    if (n.path("id").asText().equals(nodeId)) {
                        String ip = n.path("ip").asText();
                        int port = n.path("port").asInt(9090);
                        return ip + ":" + port;
                    }
                }
            }
        } catch (Exception e) {
            logger.warn("ManagerClient.getNodeAddress failed", e);
        }
        return null;
    }

    public JsonNode getWeather(String region) {
        try {
            Request req = new Request.Builder().url(managerApiUrl + "/api/weather?region=" + region).build();
            try (Response res = http.newCall(req).execute()) {
                if (!res.isSuccessful()) return null;
                return mapper.readTree(res.body().string());
            }
        } catch (Exception e) {
            logger.warn("ManagerClient.getWeather failed", e);
            return null;
        }
    }
}
```

## Application Entry Point

### SatelliteApp.java - Main application
```java
package interface_;

import application.*;
import domain.*;
import infrastructure.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class SatelliteApp {
    private static final Logger logger = LoggerFactory.getLogger(SatelliteApp.class);

    public static void main(String[] args) throws Exception {
        // Load configuration
        java.util.Properties cfg = new java.util.Properties();
        cfg.load(Files.newInputStream(Paths.get("resources/application.properties")));

        String nodeId = cfg.getProperty("node.id", "sat1");
        String nodeIp = cfg.getProperty("node.ip", "127.0.0.1");
        int nodePort = Integer.parseInt(cfg.getProperty("node.port", "9090"));

        String firebaseKey = cfg.getProperty("firebase.serviceAccountPath");
        String firebaseDb = cfg.getProperty("firebase.databaseUrl");
        String managerUrl = cfg.getProperty("manager.api.url", "http://localhost:8081");

        int bufferMax = Integer.parseInt(cfg.getProperty("buffer.maxSize", "500"));
        long reportInterval = Long.parseLong(cfg.getProperty("state.report.interval.ms", "5000"));

        // Initialize infrastructure
        FirebaseReporter reporter = new FirebaseReporter(firebaseKey, firebaseDb);
        ManagerClient managerClient = new ManagerClient(managerUrl);

        // Initialize domain
        RoutingTable routingTable = new RoutingTable();
        // Example route - in production fetch from Manager
        routingTable.addRoute("clientB", "ground1");

        // Initialize application services
        QoSModule qos = new QoSModule();
        TcpSender sender = new TcpSender(managerClient);
        BufferManager buffer = new BufferManager(bufferMax, sender);
        Forwarder forwarder = new Forwarder(nodeId, routingTable, qos, buffer, sender, reporter);

        // Start TCP server
        TcpServer server = new TcpServer(forwarder);
        server.start(nodePort);

        // Setup node info and links
        NodeInfo nodeInfo = new NodeInfo();
        nodeInfo.setId(nodeId);
        nodeInfo.setIp(nodeIp);
        nodeInfo.setActive(true);
        nodeInfo.setBandwidth(1000);
        nodeInfo.setLoad(0.1);
        nodeInfo.setCoords(new Coordinates(1200, 800, 2000));

        LinkStatus link1 = new LinkStatus();
        link1.setSrcNode(nodeId);
        link1.setDstNode("ground1");
        link1.setLatencyMs(40);
        link1.setLossRate(0.001);
        link1.setCapacityMbps(500);
        link1.setActive(true);
        link1.setCurrentLoad(0.1);

        // Start state reporting
        StateManager stateManager = new StateManager(reporter, reportInterval);
        stateManager.startPeriodic(nodeInfo, new LinkStatus[]{link1});

        // Start weather polling for QoS updates
        ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
        scheduler.scheduleAtFixedRate(() -> {
            try {
                var weatherJson = managerClient.getWeather("Asia-Pacific");
                if (weatherJson != null) {
                    double latencyFactor = weatherJson.path("latencyFactor").asDouble(1.0);
                    double lossFactor = weatherJson.path("lossFactor").asDouble(0.0);
                    qos.updateWeatherFactors(latencyFactor, lossFactor);
                }
            } catch (Exception e) {
                logger.warn("Weather poll failed", e);
            }
        }, 0, 5, TimeUnit.SECONDS);

        // Graceful shutdown
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                server.stop();
                buffer.shutdown();
                stateManager.stop();
                reporter.shutdown();
                scheduler.shutdownNow();
            } catch (Exception ignored) {}
        }));

        logger.info("Satellite service {} started on {}:{}", nodeId, nodeIp, nodePort);
    }
}
```

## Setup và Deployment

### 1. Prerequisites
```bash
# Java 17+
java -version

# Maven 3.6+
mvn -version

# Firebase project setup
# - Create project at https://console.firebase.google.com
# - Enable Realtime Database
# - Generate service account key JSON
```

### 2. Build và Package
```bash
# Clone repository
git clone <your-repo-url>
cd satellite-service

# Build with dependencies
mvn clean compile assembly:single

# Or standard build
mvn clean package
```

### 3. Configuration
```bash
# Copy và edit configuration
cp resources/application.properties.example resources/application.properties

# Update paths
vim resources/application.properties
```

### 4. Run Service
```bash
# Development mode
mvn exec:java -Dexec.mainClass="interface_.SatelliteApp"

# Production mode
java -cp target/satellite-service-0.1.0-jar-with-dependencies.jar interface_.SatelliteApp
```

## Firebase Schema

Service ghi dữ liệu theo schema sau:

```json
{
  "logs": {
    "logId": {
      "packetId": "uuid-string",
      "src": "clientA", 
      "dst": "clientB",
      "path": ["clientA", "sat1", "ground1", "clientB"],
      "timestamp": 1640995200000,
      "status": "in-flight|dropped|buffered",
      "delayMs": 45.2,
      "lossRate": 0.001
    }
  },
  "nodes": {
    "sat1": {
      "id": "sat1",
      "type": "satellite",
      "ip": "10.0.0.2",
      "coords": {"x": 1200, "y": 800, "z": 2000},
      "status": "active|inactive",
      "bandwidth": 1000,
      "load": 0.1,
      "lastUpdated": 1640995200000
    }
  },
  "links": {
    "sat1_ground1": {
      "srcNode": "sat1",
      "dstNode": "ground1", 
      "latencyMs": 40,
      "lossRate": 0.001,
      "capacityMbps": 500,
      "status": "up|down",
      "currentLoad": 0.1,
      "lastUpdated": 1640995200000
    }
  }
}
```

## API Integration

### Manager API Endpoints

**GET /api/nodes** - Lấy danh sách nodes
```json
[
  {"id":"sat1","ip":"10.0.0.2","port":9090},
  {"id":"ground1","ip":"10.0.0.5","port":9091}
]
```

**GET /api/weather?region=Asia-Pacific** - Lấy weather factors
```json
{
  "latencyFactor": 1.2,
  "lossFactor": 0.05,
  "region": "Asia-Pacific",
  "timestamp": 1640995200000
}
```

### TCP Packet Protocol

Packets được gửi qua TCP dưới dạng JSON, mỗi packet một dòng:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "src": "clientA",
  "dst": "clientB", 
  "payloadSize": 1024,
  "ttl": 8,
  "currentNode": "sat1",
  "nextHop": "ground1",
  "createdAt": 1640995200000,
  "lastUpdated": 1640995205000,
  "delayMs": 45.2,
  "lossRate": 0.001,
  "dropped": false
}
```

## Architecture và Design Patterns

### Clean Architecture Layers
- **Domain**: Pure business models, no external dependencies
- **Application**: Business logic, orchestration, policies
- **Infrastructure**: External services, I/O, databases
- **Interface**: Entry points, controllers, adapters

### Key Design Patterns
- **Pipeline Pattern**: Forwarder xử lý packet theo stages
- **Observer Pattern**: Firebase async logging
- **Strategy Pattern**: QoS modules có thể swap
- **Buffer Pattern**: Packet buffering với background flush
- **Repository Pattern**: RoutingTable abstraction

### Thread Safety
- `ConcurrentHashMap` cho routing table
- `AtomicReference` cho weather factors
- `ConcurrentLinkedQueue` cho packet buffer
- Non-blocking Firebase writes

## Testing Strategy

### Unit Tests
```bash
# Run all tests
mvn test

# Specific test class
mvn test -Dtest=BufferManagerTest

# With coverage
mvn test jacoco:report
```

### Integration Tests
```bash
# With testcontainers for Firebase emulator
mvn test -Dtest=IntegrationTest

# End-to-end packet flow
mvn test -Dtest=PacketFlowTest
```

### Performance Tests
```bash
# Load testing
mvn test -Dtest=LoadTest -Dpackets.per.second=1000

# Memory usage
mvn test -Dtest=MemoryTest -Xmx512m
```

## Monitoring và Troubleshooting

### Logging Levels
```properties
# In application.properties
org.slf4j.simpleLogger.defaultLogLevel=info
org.slf4j.simpleLogger.log.application.Forwarder=debug
org.slf4j.simpleLogger.log.infrastructure.FirebaseReporter=warn
```

### Key Metrics tại Firebase
- Packet throughput (logs/second)
- Drop rate (dropped vs total)
- Average latency per route
- Buffer utilization
- Node status transitions

### Common Issues

**High packet loss:**
```bash
# Check weather factors
curl "http://manager:8081/api/weather?region=Asia-Pacific"

# Monitor QoS module
grep "updateWeatherFactors" logs/satellite.log
```

**Buffer overflow:**
```bash
# Check buffer size
grep "Buffer full" logs/satellite.log

# Adjust buffer size
echo "buffer.maxSize=1000" >> resources/application.properties
```

**Firebase connection issues:**
```bash
# Validate service account
firebase auth:test path/to/service-account.json

# Check database rules
firebase database:get /
```

## Production Deployment

### Docker
```dockerfile
FROM openjdk:17-jre-slim

WORKDIR /app
COPY target/satellite-service-0.1.0-jar-with-dependencies.jar app.jar
COPY resources/ resources/

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:9090/health || exit 1

CMD ["java", "-jar", "app.jar"]
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: satellite-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: satellite-service
  template:
    metadata:
      labels:
        app: satellite-service
    spec:
      containers:
      - name: satellite-service
        image: satellite-service:latest
        ports:
        - containerPort: 9090
        env:
        - name: NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: FIREBASE_SERVICE_ACCOUNT_PATH
          value: /secrets/firebase-key.json
        volumeMounts:
        - name: firebase-secret
          mountPath: /secrets
        - name: config
          mountPath: /app/resources
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          tcpSocket:
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          tcpSocket:
            port: 9090
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: firebase-secret
        secret:
          secretName: firebase-service-account
      - name: config
        configMap:
          name: satellite-config
---
apiVersion: v1
kind: Service
metadata:
  name: satellite-service
spec:
  selector:
    app: satellite-service
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
```

### Security Considerations
- Firebase service account key in Kubernetes secrets
- Network policies for pod-to-pod communication
- TLS termination at ingress level
- Resource limits và quotas
- RBAC cho service accounts

## Performance Tuning

### JVM Tuning
```bash
# Production JVM args
java -Xms256m -Xmx512m \
     -XX:+UseG1GC \
     -XX:MaxGCPauseMillis=100 \
     -XX:+PrintGCDetails \
     -jar app.jar
```

### Buffer Sizing
```properties
# High throughput
buffer.maxSize=2000
buffer.flushInterval=100

# Low latency
buffer.maxSize=100
buffer.flushInterval=50
```

### Thread Pool Sizing
```java
// In TcpServer.java
ExecutorService workerPool = Executors.newFixedThreadPool(
    Runtime.getRuntime().availableProcessors() * 2
);
```

## Roadmap và Extensions

### Phase 2 Features
- [ ] gRPC communication thay vì raw TCP
- [ ] Prometheus metrics export
- [ ] Circuit breaker cho external calls
- [ ] Advanced routing algorithms (shortest path, load balancing)

### Phase 3 Features
- [ ] Machine learning QoS prediction
- [ ] Multi-path routing
- [ ] Real-time congestion control
- [ ] Edge computing integration

---

**Contact**: Development Team  
**Version**: 0.1.0  
**Last Updated**: September 2025