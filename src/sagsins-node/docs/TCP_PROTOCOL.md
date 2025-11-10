# TCP Length-Prefix Protocol Documentation

## Overview

The SAGINS Node application uses a **length-prefix protocol** for TCP communication between nodes and between nodes and users. This protocol ensures reliable message framing over TCP streams.

## Protocol Specification

### Message Format

Each message consists of two parts:

```
[4-byte length prefix][N bytes of JSON data]
```

1. **Length Prefix** (4 bytes): Big-endian integer specifying the size of the JSON payload
2. **Payload** (N bytes): JSON-serialized `Packet` object

### Byte Order (Endianness)

The protocol uses **big-endian** (network byte order) for the 4-byte length prefix:

```
Byte 0: MSB (bits 24-31)
Byte 1: bits 16-23
Byte 2: bits 8-15
Byte 3: LSB (bits 0-7)
```

This is the standard network byte order as defined in RFC 1700 and is compatible with:
- Java's `DataInputStream.readInt()`
- Java's `DataOutputStream.writeInt()`
- Most network protocols

### Maximum Packet Size

- **Maximum allowed packet size**: 16,384 bytes (16 KB)
- Packets exceeding this size will be rejected to prevent memory exhaustion attacks
- Minimum packet size: 1 byte

## Implementation

### Sender (TCP_Service)

The sender serializes packets and sends them with a length prefix:

```java
// 1. Serialize packet to JSON
byte[] packetData = objectMapper.writeValueAsBytes(packet);

// 2. Validate packet size
if (packetData.length <= 0 || packetData.length > MAX_PACKET_SIZE) {
    // Reject packet
}

// 3. Create 4-byte big-endian length prefix
byte[] lengthPrefix = new byte[] {
    (byte) (packetData.length >> 24),  // MSB
    (byte) (packetData.length >> 16),
    (byte) (packetData.length >> 8),
    (byte) packetData.length           // LSB
};

// 4. Send length prefix + packet data
outputStream.write(lengthPrefix);
outputStream.write(packetData);
outputStream.flush();
```

### Receiver (NodeGateway)

The receiver reads the length prefix first, then reads exactly that many bytes:

```java
// 1. Read 4-byte length prefix
byte[] lengthBytes = new byte[4];
dis.readFully(lengthBytes);

// 2. Convert to integer (big-endian)
int packetLength = ((lengthBytes[0] & 0xFF) << 24) |
                   ((lengthBytes[1] & 0xFF) << 16) |
                   ((lengthBytes[2] & 0xFF) << 8) |
                   (lengthBytes[3] & 0xFF);

// 3. Validate packet length
if (packetLength <= 0 || packetLength > MAX_PACKET_SIZE) {
    // Reject and close connection
}

// 4. Read exactly packetLength bytes
byte[] data = new byte[packetLength];
dis.readFully(data);

// 5. Deserialize JSON
Packet packet = objectMapper.readValue(data, Packet.class);
```

## Error Detection

The implementation includes comprehensive error detection and diagnostics:

### 1. Protocol Violation Detection

If a client sends JSON directly without a length prefix, the system will detect it:

```
ERROR: PROTOCOL ERROR detected! Received what looks like JSON data '{"pa' (0x7B227061)
instead of a length prefix. Client is NOT using length-prefix protocol.
```

Example: The error value `2065854561` (0x7B227061) = ASCII `{"pa` indicates JSON was sent directly.

### 2. Byte Order Issue Detection

If a client uses little-endian instead of big-endian:

```
ERROR: BYTE ORDER ISSUE detected! Received invalid packet length 262144 (0x00040000).
Reversed byte order would give valid length 1024. Client may be using wrong endianness.
```

### 3. Size Validation

Packets that are too large or negative:

```
WARN: Received invalid packet length 20480 (0x00005000). Expected range: 1-16384 bytes.
```

## Common Issues and Solutions

### Issue 1: "Invalid packet length 2065854561"

**Cause**: Client is sending JSON data directly without the 4-byte length prefix.

**Solution**: Ensure the client properly implements the length-prefix protocol:
1. Serialize the packet to JSON
2. Calculate the length
3. Send the 4-byte big-endian length prefix first
4. Then send the JSON data

### Issue 2: "BYTE ORDER ISSUE detected"

**Cause**: Client is using little-endian byte order instead of big-endian.

**Solution**: Use big-endian (network byte order) for the length prefix. In Java, use:
- `DataOutputStream.writeInt()` to write
- `DataInputStream.readInt()` to read

### Issue 3: "Packet size exceeds limit"

**Cause**: Packet is larger than 16 KB after serialization.

**Solution**: 
1. Reduce packet payload size
2. Split large data into multiple packets
3. Remove unnecessary fields from the packet

## Testing

The protocol implementation is tested in:
- `TCPLengthPrefixProtocolTest.java`: Protocol-level tests
- `PacketLengthValidationTest.java`: Validation and error detection tests
- `TCPCommunicationIntegrationTest.java`: End-to-end integration tests

## References

- RFC 1700: Assigned Numbers (Network Byte Order)
- Java DataInputStream/DataOutputStream documentation
- TCP Framing: https://en.wikipedia.org/wiki/Frame_(networking)
