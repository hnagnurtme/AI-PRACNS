# RL-Router: Reinforcement Learning Network Router

A sophisticated network routing system that uses reinforcement learning (DQN) to optimize packet routing through satellite and ground station networks. The system supports both RL-based and Dijkstra-based routing algorithms.

## System Architecture

```
Client (User) → RL-Router (TCPReceiver) → Database (MongoDB)
                       ↓
                  Use RL flag?
                  ↙         ↘
       Integrated DQN Model    Dijkstra Service
       (Loaded in-memory)      (DB-based graph)
                  ↓         ↓
              Next Hop Decision
                       ↓
         Fetch Full Node Data from DB
                       ↓
        Create Hop Record (All DB Fields)
                       ↓
              Forward to Next Node
                       ↓
           Destination Station Reached
                       ↓
           Deliver to Destination User
```

## Components

### 1. Database Layer (`python/utils/db_connector.py`)

**MongoConnector** - Manages all database operations:
- **Node Operations:**
  - `get_node(node_id)` - Fetch single node by ID
  - `get_all_nodes()` - Fetch all nodes
  - `get_nodes(node_ids)` - Batch fetch multiple nodes
  - `get_node_neighbors(node_id)` - Get neighbor nodes

- **User Operations:**
  - `get_user(user_id)` - Fetch user by ID
  - `get_user_by_city(city_name)` - Fetch user by city
  - `get_all_users()` - Fetch all users
  - `insert_user(user_data)` - Add new user

### 2. Model Layer

#### Node Model (`model/Node.py`)
Represents network nodes (satellites and ground stations) with:
- Position (latitude, longitude, altitude)
- Communication specs (IP, port, bandwidth, range)
- Status (operational, battery, load)
- Neighbors list

#### User Model (`model/User.py`)
Represents end users with:
- User ID and name
- City location
- IP address and port

#### Packet Model (`model/Packet.py`)
Contains routing packets with:
- Source and destination user IDs
- Station source and destination
- `use_rl` flag (true = RL routing, false = Dijkstra)
- Hop records (detailed path history)
- QoS requirements
- Analysis data (metrics)

### 3. Routing Services

#### TCPReceiver (`service/TCPReciever.py`)
**Main routing engine** that:
1. Receives packets from network nodes
2. Checks `use_rl` flag
3. Routes to appropriate service:
   - RL: Queries RL Inference Service
   - Dijkstra: Uses DijkstraService
4. Fetches node data from database
5. Creates hop records with position, distance, latency
6. Forwards packet to next hop
7. Delivers to user when destination reached

**Key Methods:**
- `process_and_forward_packet(packet)` - Main routing logic
- `get_rl_next_hop(packet)` - Query RL service
- `get_dijkstra_next_hop(packet)` - Use Dijkstra
- `deliver_to_user(packet)` - Final delivery
- `calculate_final_metrics(packet)` - Compute statistics

#### DijkstraService (`service/DijkstraService.py`)
Implements shortest path routing:
- Builds graph from database nodes
- Uses neighbor relationships
- Calculates 3D Euclidean distances
- Returns optimal path

#### Integrated RL Engine (in `TCPReciever.py`)
**No external service needed!** The RL model is loaded directly into TCPReceiver:
- Loads DQN model from checkpoint at startup
- Uses StateBuilder to create state vectors from database
- Runs inference in-memory (no TCP calls)
- Prevents routing loops automatically
- Falls back to random neighbor if model unavailable

**Benefits:**
- ⚡ Faster routing (no network overhead)
- 🔧 Simpler deployment (single service)
- 🛡️ More reliable (no external dependencies)
- 💾 Lower latency (in-memory inference)

### 4. Utilities

#### StateBuilder (`python/utils/state_builder.py`)
Creates state vectors for RL model:
- Self state (14 features): battery, queue, utilization, etc.
- Destination state (8 features): position, type, load
- Neighbor states (10 × 14 features): sorted by distance to dest

## Data Flow

### 1. Packet Initialization
```python
# Client creates packet
packet = Packet(
    source_user_id="user_A",
    destination_user_id="user_B",
    station_source="GS_HANOI",
    station_dest="GS_TOKYO",
    use_rl=True,  # Use RL routing
    ttl=20
)
```

### 2. Routing Process

For each hop:

1. **TCPReceiver** receives packet
2. Checks if at destination station
3. If not, determines next hop:
   - **RL Mode**: Queries RL service with packet state
   - **Dijkstra Mode**: Calculates shortest path
4. Fetches **from_node** and **to_node** from **MongoDB**
5. Creates **HopRecord** with:
   - Node positions (from DB)
   - Distance (calculated from positions)
   - Latency (propagation + processing + jitter)
   - Buffer state (from DB)
   - Routing algorithm used
6. Adds hop record to packet
7. Forwards to next node

### 3. Final Delivery

When packet reaches destination station:
1. Looks up destination user in database
2. Calculates final metrics:
   - Total latency
   - Total distance
   - Average latency per hop
   - Number of hops
3. Delivers to user's IP:port
4. Prints analysis results

## Database Schema

### Nodes Collection
```javascript
{
  "nodeId": "GS_HANOI",
  "nodeName": "Ground Station Hanoi",
  "nodeType": "GROUND_STATION",
  "position": {
    "latitude": 21.0285,
    "longitude": 105.8542,
    "altitude": 0.0
  },
  "communication": {
    "ipAddress": "10.0.0.1",
    "port": 7700,
    "maxRangeKm": 2000.0,
    ...
  },
  "isOperational": true,
  "batteryChargePercent": 95,
  "nodeProcessingDelayMs": 2.5,
  "resourceUtilization": 0.35,
  "neighbors": ["LEO-01", "LEO-02", "MEO-01"]
}
```

### Users Collection
```javascript
{
  "userId": "user-Singapore",
  "userName": "User_Singapore",
  "cityName": "Singapore",
  "ipAddress": "127.0.0.1",
  "port": 10000
}
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install pymongo torch numpy python-dotenv
```

### 2. Start MongoDB
```bash
# Local MongoDB
docker run -d -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=user \
  -e MONGO_INITDB_ROOT_PASSWORD=password123 \
  mongo:latest
```

### 3. Initialize Database
```python
# Populate nodes
from model.Node import generate_rl_test_data, save_nodes_to_db
nodes = generate_rl_test_data()
save_nodes_to_db(nodes)

# Add users
from python.utils.db_connector import MongoConnector
db = MongoConnector()
users = [
    {
        "userId": "user-Hanoi",
        "userName": "User_Hanoi",
        "cityName": "Hanoi",
        "ipAddress": "127.0.0.1",
        "port": 10001
    },
    {
        "userId": "user-Tokyo",
        "userName": "User_Tokyo",
        "cityName": "Tokyo",
        "ipAddress": "127.0.0.1",
        "port": 10002
    }
]
db.clear_and_insert_users(users)
```

### 4. Start Services

**Terminal 1 - TCP Receiver (Router with Integrated RL):**
```bash
cd src/rl-router
python service/TCPReciever.py
# Listens on port 65432
# RL Model: Loaded ✅ (or Not Available ⚠️)
```

**Terminal 2 - User Receiver (Destination - Optional):**
```bash
cd src/rl-router
python UserTestDest.py  # Example user receiver
```

**Terminal 3 - Send Test Packet:**
```bash
cd src/rl-router
python service/TCPSender.py
```

**That's it!** Only ONE service needed for routing. The RL model is integrated directly.

## Testing

### Test with RL Routing
```python
from service.TCPSender import create_test_packet, send_packet

# Create RL packet
packet = create_test_packet(use_rl=True)
send_packet(packet, 'localhost', 65432)
```

### Test with Dijkstra Routing
```python
packet = create_test_packet(use_rl=False)
send_packet(packet, 'localhost', 65432)
```

## Key Features

### 1. Dual Routing Algorithms
- **RL (DQN)**: Learns optimal paths considering network state
- **Dijkstra**: Classic shortest path algorithm

### 2. Database Integration
- All node and user data stored in MongoDB
- Real-time fetching of node states
- Scalable architecture

### 3. Detailed Hop Records
Each hop includes:
- Node positions (lat/lon/alt)
- 3D distance calculation
- Propagation delay (speed of light)
- Processing delay (from node specs)
- Buffer state (queue size, utilization)
- Algorithm used

### 4. User Delivery
- Automatic user lookup from database
- Delivery to user's IP:port
- Complete metrics calculation

### 5. Network Metrics
- Total/average latency
- Total/average distance
- Hop count
- Success rate
- Full path history

## Configuration

### Environment Variables
```bash
# .env file
MONGODB_URI=mongodb://user:password123@localhost:27017
```

### Router Configuration
```python
# service/TCPReciever.py
router = TCPReceiver(
    host='localhost',
    port=65432,
    model_path="models/checkpoints/dqn_checkpoint_fullpath_latest.pth"
)
```

**RL Model Path:**
- Default: `models/checkpoints/dqn_checkpoint_fullpath_latest.pth`
- If not found, router uses fallback routing (random neighbor)
- Model is loaded once at startup for fast inference

## Architecture Highlights

### Separation of Concerns
- **Database Layer**: Pure data access
- **Model Layer**: Domain objects
- **Service Layer**: Business logic
- **Transport Layer**: TCP communication

### Extensibility
- Easy to add new routing algorithms
- Pluggable database backends
- Configurable RL models

### Performance
- Batch database queries
- Efficient state vector construction
- Minimal network overhead

## Troubleshooting

### Issue: RL model not loaded
- Check if model file exists: `models/checkpoints/dqn_checkpoint_fullpath_latest.pth`
- Verify PyTorch is installed: `pip install torch`
- Check startup message: "RL Model: Loaded ✅" or "Not Available ⚠️"
- Router will automatically fall back to random neighbor selection if model unavailable

### Issue: Database connection failed
- Verify MongoDB is running
- Check credentials in MONGODB_URI
- Ensure database name is correct (sagsin_network)

### Issue: No path found
- Verify nodes have neighbor relationships
- Check if nodes are operational
- Ensure source and dest stations exist in database

## Example Output

```
--- Received Packet ---
Packet ID: a1b2c3d4-5678-90ab-cdef-1234567890ab
From: user_A (Station: GS_HANOI)
To: user_B (Station: GS_TOKYO)
Current Holder: GS_HANOI
Path History: ['GS_HANOI']
Using RL: True

--- Using RL for routing ---
Next hop for packet a1b2c3d4... is LEO-05 (using REINFORCEMENT_LEARNING)
Forwarding packet to LEO-05 at 10.1.0.5:7805

...

--- Packet Reached Destination Station ---
Delivering packet to user User_Tokyo at 127.0.0.1:10002

--- Calculating Final Metrics ---
Final Analysis:
  Total Latency: 234.56 ms
  Total Distance: 12345.67 km
  Number of Hops: 8
  Average Latency per Hop: 29.32 ms
  Average Distance per Hop: 1543.21 km
  Path: GS_HANOI -> LEO-05 -> LEO-12 -> MEO-03 -> LEO-23 -> LEO-31 -> GS_TOKYO
  Algorithm: RL
```

## Future Enhancements

1. **Multi-threaded routing** for concurrent packets
2. **Caching** of frequently used paths
3. **Real-time node status updates** via websockets
4. **Advanced QoS** enforcement
5. **Path visualization** dashboard
6. **Distributed RL training** from live traffic

## Contributing

When adding new features:
1. Update database schema if needed
2. Add corresponding model updates
3. Update service logic
4. Add tests
5. Update this README

## License

[Your License Here]
