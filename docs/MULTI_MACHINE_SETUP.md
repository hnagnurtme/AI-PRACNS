# Multi-Machine Network Setup Guide

This guide explains how to set up and run the SAGSINS network simulation across multiple machines, enabling scenarios like:
- Client in Da Nang → Node in Da Nang → Node in Hue → Client in Hue

## 🌐 Overview

The system now automatically detects LAN IP addresses and updates the database on startup, making multi-machine deployment much simpler.

## ⚙️ Automatic IP Detection

### How It Works

Both nodes and clients automatically detect their LAN IP address using the following strategy:

1. **Network Interface Enumeration**: Scans all network interfaces on the machine
2. **IPv4 Preference**: Prioritizes IPv4 addresses over IPv6
3. **Filtering**: Skips loopback (127.0.0.1) and link-local (169.254.x.x) addresses
4. **Fallback**: If no suitable interface is found, falls back to `InetAddress.getLocalHost()`

### Node Startup

When a node starts (`SimulationMain`):
```bash
🔍 Detecting LAN IP address...
Found IPv4 LAN IP: 192.168.1.10 on interface eth0
✅ LAN IP detected from network interfaces: 192.168.1.10
📝 Updating node IP addresses to 192.168.1.10...
✅ Node IP addresses flushed to database.
```

### Client Startup

When a client starts listening (`MainController`):
```bash
🌐 Auto-detected LAN IP: 192.168.1.100
✅ Updated user USER-DANANG IP: 127.0.0.1 → 192.168.1.100
Listening on 192.168.1.100:8001
```

## 🖥️ Multi-Machine Setup

### Prerequisites

- All machines must be on the same LAN (or have network connectivity)
- MongoDB must be accessible from all machines
- Firewall rules must allow TCP connections on node and client ports

### Step 1: Configure MongoDB

Ensure MongoDB is accessible from all machines. You have two options:

**Option A: Shared MongoDB Instance**
```bash
# On the MongoDB server machine
# Edit mongod.conf to bind to all interfaces
bindIp: 0.0.0.0
```

**Option B: MongoDB Connection String**
```bash
# Set environment variable on each machine
export MONGODB_URI="mongodb://mongo-server:27017/sagsins"
```

### Step 2: Deploy Nodes

#### Machine 1 (Da Nang Node - 192.168.1.10)

```bash
cd src/sagsins-node
mvn clean compile

# Optional: Override auto-detected IP
export NODE_HOST_IP=192.168.1.10

# Start the node
mvn exec:java -Dexec.mainClass="com.sagin.util.SimulationMain"
```

#### Machine 2 (Hue Node - 192.168.1.20)

```bash
cd src/sagsins-node
mvn clean compile

# Optional: Override auto-detected IP
export NODE_HOST_IP=192.168.1.20

# Start the node
mvn exec:java -Dexec.mainClass="com.sagin.util.SimulationMain"
```

### Step 3: Deploy Clients

#### Machine 3 (Da Nang Client - 192.168.1.100)

```bash
cd src/client-simulator
mvn clean compile

# Start the JavaFX client
mvn exec:java -Dexec.mainClass="com.example.MainApp"
```

**In the UI:**
1. Select sender: `USER-DANANG`
2. Click "Start Listening" - IP will auto-update to 192.168.1.100
3. Select destination: `USER-HUE`
4. Click "Send" to transmit packets

#### Machine 4 (Hue Client - 192.168.1.200)

```bash
cd src/client-simulator
mvn clean compile

# Start the JavaFX client
mvn exec:java -Dexec.mainClass="com.example.MainApp"
```

**In the UI:**
1. Select sender: `USER-HUE`
2. Click "Start Listening" - IP will auto-update to 192.168.1.200
3. Wait to receive packets from Da Nang

## 🔍 Verification

### Check Node IPs in Database

```javascript
// Connect to MongoDB
use sagsins

// Check node IP addresses
db.nodes.find({}, {nodeId: 1, "communication.ipAddress": 1, "communication.port": 1})
```

Expected output:
```json
{ "nodeId": "NODE-DANANG", "communication": { "ipAddress": "192.168.1.10", "port": 7001 }}
{ "nodeId": "NODE-HUE", "communication": { "ipAddress": "192.168.1.20", "port": 7002 }}
```

### Check User IPs in Database

```javascript
// Check user IP addresses
db.users.find({}, {userId: 1, ipAddress: 1, port: 1})
```

Expected output:
```json
{ "userId": "USER-DANANG", "ipAddress": "192.168.1.100", "port": 8001 }
{ "userId": "USER-HUE", "ipAddress": "192.168.1.200", "port": 8002 }
```

### Test Connectivity

From any machine, test node connectivity:
```bash
# Test Da Nang node
telnet 192.168.1.10 7001

# Test Hue node
telnet 192.168.1.20 7002
```

## 📊 Packet Flow Example

**Scenario**: User in Da Nang sends packet to User in Hue

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ USER-DANANG     │────▶│ NODE-DANANG      │────▶│ NODE-HUE         │────▶│ USER-HUE        │
│ 192.168.1.100   │     │ 192.168.1.10     │     │ 192.168.1.20     │     │ 192.168.1.200   │
│ Port 8001       │     │ Port 7001        │     │ Port 7002        │     │ Port 8002       │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └─────────────────┘
   Machine 3               Machine 1                Machine 2                Machine 4
```

**Log Output**:
```
[NODE-DANANG] 📥 Nhận Packet PKT-001 tại NODE-DANANG | TTL: 15
[NODE-DANANG] 🔄 Routing Packet PKT-001 | NODE-DANANG → NODE-HUE (next: NODE-HUE)
[NODE-HUE]    📥 Nhận Packet PKT-001 tại NODE-HUE | TTL: 14
[NODE-HUE]    ✅ Packet PKT-001 reached destination station NODE-HUE
[NODE-HUE]    Forwarding to user USER-HUE
[USER-HUE]    ✅ Received packet PKT-001
```

## 🐛 Troubleshooting

### IP Detection Issues

**Problem**: Wrong IP detected (e.g., Docker bridge IP, VPN IP)

**Solution**: Override with environment variable
```bash
export NODE_HOST_IP=192.168.1.10  # For nodes
```

For clients, the IP is detected when "Start Listening" is clicked. If wrong:
1. Edit the database directly:
```javascript
db.users.updateOne(
  {userId: "USER-DANANG"},
  {$set: {ipAddress: "192.168.1.100"}}
)
```
2. Restart the client and click "Start Listening" again

### Connection Refused

**Problem**: `java.net.ConnectException: Connection refused`

**Possible Causes**:
1. Node not running on target machine
2. Firewall blocking port
3. Wrong IP address in database

**Solution**:
```bash
# Check if node is listening
netstat -an | grep 7001

# Check firewall (Ubuntu/Debian)
sudo ufw status
sudo ufw allow 7001/tcp

# Check firewall (CentOS/RHEL)
sudo firewall-cmd --list-all
sudo firewall-cmd --add-port=7001/tcp --permanent
sudo firewall-cmd --reload
```

### Database Connection Issues

**Problem**: `MongoTimeoutException` or connection errors

**Solution**:
```bash
# Check MongoDB is running
sudo systemctl status mongod

# Check MongoDB allows remote connections
# Edit /etc/mongod.conf
bindIp: 0.0.0.0

# Restart MongoDB
sudo systemctl restart mongod

# Test connection from client machine
mongo mongodb://192.168.1.5:27017/sagsins
```

### Packet Not Delivered

**Problem**: Packet sent but never reaches destination

**Debugging Steps**:

1. **Check routing tables in database**:
```javascript
db.routing_tables.find({sourceNodeId: "NODE-DANANG", destinationNodeId: "NODE-HUE"})
```

2. **Check node health**:
```javascript
db.nodes.find({}, {nodeId: 1, operational: 1, healthy: 1})
```

3. **Enable detailed logging** (add to application.properties or logback.xml):
```properties
logging.level.com.sagin.network=DEBUG
logging.level.com.sagin.routing=DEBUG
```

4. **Check TTL**: If packets traverse many hops, TTL might expire
- Default TTL is 15 hops
- Increase if needed in packet creation

## 🔒 Security Considerations

### Network Security

- All communication is currently over plain TCP (no encryption)
- For production, consider:
  - TLS/SSL for TCP connections
  - VPN for multi-site deployments
  - Firewall rules to restrict access

### Database Security

- MongoDB should be protected with authentication
- Use connection strings with credentials:
```bash
export MONGODB_URI="mongodb://username:password@192.168.1.5:27017/sagsins?authSource=admin"
```

## 📈 Performance Tips

### Network Latency

- Use wired connections instead of WiFi when possible
- Keep nodes and MongoDB on same subnet if possible
- Monitor network latency: `ping 192.168.1.10`

### MongoDB Performance

- Create indexes for frequently queried fields:
```javascript
db.nodes.createIndex({"nodeId": 1})
db.users.createIndex({"userId": 1})
db.routing_tables.createIndex({"sourceNodeId": 1, "destinationNodeId": 1})
```

### Resource Limits

- Each node creates thread pools for handling connections
- Monitor CPU and memory usage
- Adjust JVM heap size if needed:
```bash
export MAVEN_OPTS="-Xmx4g -Xms2g"
```

## 📚 Additional Resources

- [MongoDB Remote Access Guide](https://www.mongodb.com/docs/manual/tutorial/configure-network-access/)
- [Java Network Programming Best Practices](https://docs.oracle.com/javase/tutorial/networking/)
- [TCP/IP Networking Fundamentals](https://www.ietf.org/rfc/rfc793.txt)

## ✅ Verification Checklist

Before running multi-machine simulations:

- [ ] MongoDB is accessible from all machines
- [ ] All machines can ping each other
- [ ] Required ports are open in firewalls (7001-7010, 8001-8010)
- [ ] Node startup logs show correct IP detection
- [ ] Client UI shows correct IP after "Start Listening"
- [ ] Database contains correct IP addresses for all nodes/users
- [ ] Simple telnet test succeeds to all node ports
- [ ] Test packet successfully delivered end-to-end
