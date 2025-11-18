# BatchPacketService Documentation

## Overview

The `BatchPacketService` automatically saves packets to MongoDB in two collections when packets arrive at the TCP receiver, whether they are successfully delivered or dropped.

## Architecture

### Collections

1. **`two_packets`**: Stores the latest pair of packets (Dijkstra + RL) for each source-destination user pair
   - **Behavior**: UPSERT (Replace existing document)
   - **PairId Format**: `sourceUserId_destinationUserId`
   - **Purpose**: Always maintain the latest state for each algorithm

2. **`batch_packets`**: Accumulates all packet pairs over time
   - **Behavior**: APPEND (Add to array)
   - **BatchId Format**: `sourceUserId_destinationUserId`
   - **Purpose**: Historical tracking of all packet transmissions

## How It Works

### Automatic Packet Saving

The service is integrated into [TCPReciever.py](../service/TCPReciever.py) and automatically triggers on two events:

1. **Packet Dropped** (in `_drop_packet()` method)
2. **Packet Delivered** (in `deliver_to_user()` method)

### Save Flow

```python
# When a packet arrives (dropped or delivered)
def save_packet(packet: Packet):
    # 1. Create pairId from source and destination users
    pair_id = f"{source_user_id}_{destination_user_id}"

    # 2. Save/Update TwoPacket (UPSERT - overwrites existing)
    #    - If Dijkstra packet: updates dijkstraPacket field
    #    - If RL packet: updates rlPacket field
    _save_two_packet(pair_id, packet)

    # 3. Append to BatchPacket (INSERT into array)
    #    - Always adds the current TwoPacket to the batch array
    _append_to_batch(pair_id, packet)
```

### Example Workflow

**Scenario**: User-Singapore sends packets to User-Hanoi

1. **Dijkstra Packet Arrives** (Delivered):
   ```
   TwoPacket Collection:
   {
     "pairId": "user-Singapore_user-Hanoi",
     "dijkstraPacket": { ... packet data ... },
     "rlPacket": null
   }

   BatchPacket Collection:
   {
     "batchId": "user-Singapore_user-Hanoi",
     "totalPairPackets": 1,
     "packets": [
       {
         "pairId": "user-Singapore_user-Hanoi",
         "dijkstraPacket": { ... },
         "rlPacket": null
       }
     ]
   }
   ```

2. **RL Packet Arrives** (Delivered):
   ```
   TwoPacket Collection (UPDATED):
   {
     "pairId": "user-Singapore_user-Hanoi",
     "dijkstraPacket": { ... packet data ... },
     "rlPacket": { ... packet data ... }  ← UPDATED
   }

   BatchPacket Collection (APPENDED):
   {
     "batchId": "user-Singapore_user-Hanoi",
     "totalPairPackets": 2,
     "packets": [
       { ... first entry ... },
       {
         "pairId": "user-Singapore_user-Hanoi",
         "dijkstraPacket": { ... },
         "rlPacket": { ... }  ← NEW ENTRY ADDED
       }
     ]
   }
   ```

3. **Another Dijkstra Packet Arrives** (Dropped):
   ```
   TwoPacket Collection (REPLACED):
   {
     "pairId": "user-Singapore_user-Hanoi",
     "dijkstraPacket": { ... LATEST dropped packet ... },  ← REPLACED
     "rlPacket": { ... previous RL packet ... }
   }

   BatchPacket Collection (APPENDED):
   {
     "batchId": "user-Singapore_user-Hanoi",
     "totalPairPackets": 3,
     "packets": [
       { ... first entry ... },
       { ... second entry ... },
       {
         "pairId": "user-Singapore_user-Hanoi",
         "dijkstraPacket": { ... dropped packet ... },
         "rlPacket": { ... }  ← NEW ENTRY ADDED
       }
     ]
   }
   ```

## Key Features

### ✅ Automatic Operation
- No manual intervention required
- Automatically saves on packet drop or delivery
- Integrated seamlessly into TCPReceiver

### ✅ Dual Collection Strategy
- **TwoPacket**: Latest state (fast lookup for current comparison)
- **BatchPacket**: Complete history (analytics and trend analysis)

### ✅ Handles Both Success and Failure
- Dropped packets are saved with `dropped: true` and `dropReason`
- Successful deliveries are saved with full path and metrics
- Both contribute to comprehensive analysis

### ✅ Pair-Based Tracking
- Each source-destination pair has unique ID
- Easy to compare Dijkstra vs RL performance
- Supports multiple simultaneous user pairs

## API Reference

### Core Methods

#### `save_packet(packet: Packet)`
Main entry point - automatically saves packet to both collections.

```python
service.save_packet(packet)
```

#### `get_two_packet(pair_id: str) -> Optional[Dict]`
Retrieve the latest TwoPacket for a user pair.

```python
two_packet = service.get_two_packet("user-Singapore_user-Hanoi")
```

#### `get_batch(batch_id: str) -> Optional[Dict]`
Retrieve complete batch history for a user pair.

```python
batch = service.get_batch("user-Singapore_user-Hanoi")
```

#### `get_batch_statistics(batch_id: str) -> Optional[Dict]`
Get performance statistics for a batch.

```python
stats = service.get_batch_statistics("user-Singapore_user-Hanoi")
# Returns:
# {
#   "batchId": "user-Singapore_user-Hanoi",
#   "totalPackets": 10,
#   "dijkstra": {
#     "successful": 7,
#     "dropped": 3,
#     "successRate": 0.7,
#     "avgLatencyMs": 145.2
#   },
#   "rl": {
#     "successful": 8,
#     "dropped": 2,
#     "successRate": 0.8,
#     "avgLatencyMs": 132.5
#   }
# }
```

## Integration Example

### In TCPReceiver

```python
class TCPReceiver:
    def __init__(self, ...):
        # Initialize BatchPacket service
        self.batch_packet_service = BatchPacketService(self.db)

    def _drop_packet(self, packet: Packet, drop_reason: str):
        packet.dropped = True
        packet.drop_reason = drop_reason

        # ... logging ...

        # ✅ AUTO-SAVE to collections
        self.batch_packet_service.save_packet(packet)

    def deliver_to_user(self, packet: Packet):
        # ... delivery logic ...

        # ✅ AUTO-SAVE to collections
        self.batch_packet_service.save_packet(packet)
```

## Testing

Run the test script to verify functionality:

```bash
cd /Users/anhnon/PBL4/src/rl-router
python test_batch_packet_service.py
```

The test creates sample packets (both successful and dropped) and verifies they are correctly saved to both collections.

## MongoDB Queries

### Find all TwoPackets
```javascript
db.two_packets.find()
```

### Find specific user pair
```javascript
db.two_packets.findOne({"pairId": "user-Singapore_user-Hanoi"})
```

### Find all batch history for a pair
```javascript
db.batch_packets.findOne({"batchId": "user-Singapore_user-Hanoi"})
```

### Count total packets in a batch
```javascript
db.batch_packets.aggregate([
  {$match: {batchId: "user-Singapore_user-Hanoi"}},
  {$project: {packetCount: {$size: "$packets"}}}
])
```

### Find all dropped packets in batch
```javascript
db.batch_packets.aggregate([
  {$match: {batchId: "user-Singapore_user-Hanoi"}},
  {$unwind: "$packets"},
  {$match: {
    $or: [
      {"packets.dijkstraPacket.dropped": true},
      {"packets.rlPacket.dropped": true}
    ]
  }}
])
```

## Benefits

1. **Performance Comparison**: Easy to compare Dijkstra vs RL algorithms
2. **Historical Analysis**: Track performance trends over time
3. **Debugging**: Identify patterns in packet drops
4. **QoS Monitoring**: Verify service quality requirements
5. **Research**: Analyze routing algorithm effectiveness

## Dependencies

- MongoDB (via MongoConnector)
- [Packet.py](../model/Packet.py) - Packet data model
- [TwoPacket.py](../model/TwoPacket.py) - Packet pair model
- [BatchPacket.py](../model/BatchPacket.py) - Batch model

## Notes

- Ensure MongoDB is running and accessible
- Database connection configured in `.env` file
- Collections are created automatically on first insert
- No manual cleanup required - service handles everything
