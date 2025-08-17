
# AI-Powered Resource Allocation in Cloud and Network Systems

## Project Overview
This project focuses on **optimizing resource allocation** in **Space–Air–Ground–Sea Integrated Networks (SAGSINs)** using **Generative AI** and heuristic algorithms. The primary goal is to minimize communication latency between network nodes (satellites, UAVs, ground/sea stations) while optimizing the use of bandwidth, energy, and other resources.

---

## System Architecture

### 1. Client
- Clients send **requests** to users at different locations.
- Clients can be static (e.g., ground stations) or mobile (e.g., UAVs, ships).
- Responsibilities:
  - Send resource allocation requests to the server.
  - Receive and process responses with optimized resource allocation plans.

### 2. Resource Allocation Server
- Receives requests from clients.
- Processes requests using **optimization algorithms** based on:
  - Client and user locations.
  - Bandwidth, latency, and current load of SAGSIN nodes (satellites, UAVs, ground/sea stations).
  - Service Level Agreements (SLAs) or user priorities.
- Stores network state and request history in **MongoDB**.
- Communicates with the AI Server for optimal allocation strategies.

### 3. AI/GenAI Server
- Collects historical network usage data from **MongoDB**.
- Trains and generates **optimal resource allocation strategies** based on network conditions and traffic forecasts.
- Returns optimization strategies to the Resource Allocation Server for real-time application.

---

## Data Flow
```plaintext
Client A
   └─ Request ─> Resource Allocation Server ─> Fetch SAGSIN node info (satellites, UAVs, stations)
   └─ Store data in MongoDB ─> AI Server trains & optimizes
   └─ Optimal strategy ─> Resource Allocation Server ─> Response ─> Client B
```

---

## Data Storage (MongoDB)
The system uses **MongoDB** to store:
- **SAGSIN Nodes**: Location, bandwidth, latency, current load.
  ```json
  {
    "nodeId": "string",
    "type": "string (satellite/UAV/ground/sea)",
    "location": { "lat": "float", "lon": "float" },
    "bandwidth": "float",
    "latency": "float",
    "load": "float",
    "timestamp": "ISODate"
  }
  ```
- **Requests**: Client, recipient user, timestamp, response time, status.
  ```json
  {
    "requestId": "string",
    "clientId": "string",
    "recipientId": "string",
    "timestamp": "ISODate",
    "responseTime": "float",
    "status": "string (pending/success/failed)"
  }
  ```
- **AI Strategies**: Optimized allocation scenarios and simulation results.
  ```json
  {
    "strategyId": "string",
    "nodes": ["nodeId"],
    "allocationPlan": { "bandwidth": "float", "path": ["nodeId"] },
    "predictedLatency": "float",
    "timestamp": "ISODate"
  }
  ```

---

## Optimization Algorithms
- **Input**: Client and user locations, SAGSIN network state.
- **Output**: Optimal path and resource allocation plan.
- **Methods**:
  - **Heuristic Algorithms**: Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO).
  - **Generative AI**: Reinforcement Learning with Generative Models to learn and improve allocation scenarios.
- **Objectives**:
  - Minimize latency.
  - Maximize throughput.
  - Balance resource utilization (bandwidth, energy).

---

## Expected Outcomes
- **Generative AI** generates multiple optimized resource allocation plans based on network conditions.
- **Reduced latency** in communication between SAGSIN nodes.
- **Improved efficiency** in bandwidth, energy, and resource utilization.
- **Real-time adaptability** through predictive models for network demand.

---

## Technologies Used
- **Programming Language**: Python (primary), with potential for Java/C++ for specific components.
- **AI/ML Frameworks**: PyTorch, TensorFlow, HuggingFace Transformers.
- **Optimization Algorithms**: PSO, ACO, Reinforcement Learning.
- **Database**: MongoDB for storing node states, requests, and AI strategies.
- **Client–Server Communication**: REST API (via FastAPI/Flask) or gRPC for high-performance communication.
- **Deployment**: Docker for containerization, Kubernetes for orchestration (optional).

---

## Project Structure
```plaintext
PBL4/
├── data/                 # Input data, request history, node states
├── models/               # Trained AI/GenAI models
├── src/                  # Source code for client, server, and optimization algorithms
│   ├── client/           # Client-side logic
│   ├── server/           # Resource Allocation Server and AI Server
│   └── optimization/      # PSO, ACO, and RL algorithms
├── scripts/              # Scripts for simulation and optimization
├── README.md             # Project documentation
└── requirements.txt      # Required Python libraries
```

---

## Setup and Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/PBL4.git
   cd PBL4
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up MongoDB**:
   - Install MongoDB locally or use a cloud-based instance (e.g., MongoDB Atlas).
   - Configure the connection string in `src/server/config.py`.

4. **Run the system**:
   - Start the Resource Allocation Server:
     ```bash
     python src/server/resource_server.py
     ```
   - Start the AI Server:
     ```bash
     python src/server/ai_server.py
     ```
   - Run a client simulation:
     ```bash
     python src/client/client.py
     ```

5. **Run simulations**:
   - Use scripts in the `scripts/` directory to simulate network conditions and test optimization algorithms:
     ```bash
     python scripts/simulate_network.py
     ```

---

## Bug Reporting
To report issues, use the provided `bug.yml` template in the repository. Include:
- System/Service name (e.g., PBL4).
- Detailed bug description.
- Steps to reproduce, expected vs. actual behavior.
- Environment details (OS, Python version, etc.).
- Logs or screenshots for debugging.

Example bug report template: [bug.yml](bug.yml).

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.