
# 🌐 AI-Powered Resource Allocation in Cloud and Network Systems

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=flat-square&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![REST API](https://img.shields.io/badge/REST-API-blue?style=flat-square)]()
[![MIT License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> A **Generative AI-based simulation** for optimizing resource allocation in **Space–Air–Ground–Sea Integrated Networks (SAGSINs)** using **heuristic algorithms** and **reinforcement learning**.
### 🧰 Tech Stack

#### Languages & Frameworks
![Java](https://img.shields.io/badge/Java-ED8B00?style=flat-square&logo=java&logoColor=white)
![Spring Boot](https://img.shields.io/badge/Spring_Boot-6DB33F?style=flat-square&logo=spring&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black)

#### AI & ML
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

#### Database & Realtime
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=flat-square&logo=firebase&logoColor=black)

---

## 🚀 Project Overview
This project focuses on **minimizing communication latency** and **optimizing bandwidth and energy usage** across integrated SAGSIN networks.  
It combines **heuristic algorithms (PSO, ACO)** with **Generative AI** to deliver intelligent resource allocation and real-time adaptability.

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
