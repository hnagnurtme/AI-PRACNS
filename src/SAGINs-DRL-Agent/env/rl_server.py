import json
import socket
import threading
import time
import os
import pickle
import sys
# Ensure repo root is on sys.path so top-level imports (utils, env) resolve when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
except Exception:
    raise ImportError("pymongo is required for RLServer to fetch live nodes from MongoDB. Install with: pip install pymongo")
try:
    # When ran as package
    from env.sagins_env import SAGINsEnv
    from env.rl_model import RLAgent
except Exception:
    # When ran as a script from the env/ directory
    from sagins_env import SAGINsEnv
    from rl_model import RLAgent
import numpy as np
import uuid

class RLServer:
    def __init__(self, host='0.0.0.0', port=5050, mongo_uri='mongodb://user:password123@localhost:27017/?authSource=admin', db_name='sagsin_network', collection_name='network_nodes', checkpoint_path='rl_checkpoint.pth', cache_file='path_cache.pkl'):
        self.host = host
        self.port = port
        # Connect to MongoDB (required for live node data)
        try:
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # force a server selection to verify connection
            self.mongo_client.server_info()
            self.db = self.mongo_client[db_name]
            self.collection = self.db[collection_name]
        except ServerSelectionTimeoutError as e:
            raise ConnectionFailure(f"Cannot connect to MongoDB at '{mongo_uri}': {e}\nPlease ensure MongoDB is running and MONGO_URI is correct.")
        except Exception as e:
            raise ConnectionFailure(f"MongoDB connection error: {e}")
        self.cache = {}  # key: (stationSource, stationDest, serviceType), value: {'id': str, 'path': list}
        self.cache_file = cache_file
        self.load_cache()
        
        self.nodes = self.get_nodes()
        if not self.nodes:
            raise ValueError("No nodes found in MongoDB collection. Please populate the collection with node documents (see docs/NODE.md for example).")
        
        # Support both 'type' and 'nodeType' keys in node documents
        sample_source = next((node_id for node_id, node in self.nodes.items() if (node.get('type') == 'GROUND_STATION' or node.get('nodeType') == 'GROUND_STATION')), list(self.nodes.keys())[0])
        sample_dest = next((node_id for node_id in self.nodes.keys() if node_id != sample_source), list(self.nodes.keys())[-1])

        dummy_packet = {
            'stationSource': sample_source,
            'stationDest': sample_dest,
            'currentHoldingNodeId': sample_source,
            'TTL': 20,  # Increase TTL for more hops
            'serviceType': 'VIDEO_STREAM',
            'payloadSizeByte': 512,
            'priorityLevel': 1,
            'accumulatedDelayMs': 0,
            'maxAcceptableLatencyMs': 150,
            'maxAcceptableLossRate': 0.01
        }
        dummy_env = SAGINsEnv(self.nodes, dummy_packet)
        state_size = len(dummy_env._get_obs())
        action_size = len(self.nodes)
        self.agent = RLAgent(state_size, action_size, checkpoint_path)
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # allow quick reuse of address during dev/testing
        try:
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"RL Server listening on {self.host}:{self.port}")
    
    def get_nodes(self):
        nodes = {}
        # Require a Mongo collection (we expect live data)
        if self.collection is None:
            raise ConnectionFailure("MongoDB collection not initialized; cannot fetch nodes.")
        try:
            for doc in self.collection.find():
                node_id = doc.get('nodeId')
                if node_id:
                    nodes[node_id] = doc
        except Exception:
            # any error reading Mongo -> raise so caller knows
            raise
        return nodes
    
    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            print("Loaded cache")
    
    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def save_checkpoint(self):
        self.agent.save_checkpoint()
        self.save_cache()
        print("Saved checkpoint and cache")
    
    def find_path_with_rl(self, env):
        return self.agent.find_path(env)
    
    def handle_client(self, client_socket):
        try:
            data = client_socket.recv(4096).decode('utf-8')
            if not data:
                return
            packet = json.loads(data)
            print(f"\n{'='*60}")
            print(f"📩 Received packet: {packet}")
            print(f"{'='*60}")
            
            source = packet['stationSource']
            dest = packet['stationDest']
            service_type = packet.get('serviceType', 'VIDEO_STREAM')
            current = packet.get('currentHoldingNodeId', source)
            packet['currentHoldingNodeId'] = current
            
            key = (source, dest, service_type)
            nodes = self.get_nodes()
            print(f"📊 Available nodes: {len(nodes)} total")
            
            if source not in nodes or dest not in nodes:
                response = {'error': 'Source or dest not found'}
            else:
                env = SAGINsEnv(nodes, packet)
                print(f"🎯 Current: {current} → Destination: {dest}, TTL: {packet['TTL']}")
                
                # Determine path (from cache or new)
                path = None
                is_cached = False
                
                if key in self.cache:
                    cached_path = self.cache[key]['path']
                    print(f"💾 Found cached path: {' → '.join(cached_path)}")
                    is_cached = True
                    
                    if current in cached_path:
                        idx = cached_path.index(current)
                        if idx + 1 < len(cached_path):
                            # Try to re-optimize remaining path
                            new_path = self.find_path_with_rl(env)
                            print(f"🔄 Re-optimized path: {new_path}")
                            
                            if new_path and new_path != cached_path[idx:]:
                                cached_path[idx + 1:] = new_path[1:]
                                self.cache[key]['path'] = cached_path
                                print(f"✅ Updated cached path: {' → '.join(cached_path)}")
                            
                            path = cached_path
                            response = {'path': cached_path, 'nextHopNodeId': cached_path[idx + 1]}
                        else:
                            path = cached_path
                            response = {'status': 'at_dest', 'path': cached_path}
                            del self.cache[key]
                    else:
                        response = {'error': 'Current not in cached path'}
                else:
                    # Find new path
                    path = self.find_path_with_rl(env)
                    print(f"🆕 Found new path: {path}")
                    
                    if path:
                        path_id = str(uuid.uuid4())
                        self.cache[key] = {'id': path_id, 'path': path}
                        self.agent.replay()
                        
                        if current in path:
                            idx = path.index(current)
                            response = {'path': path, 'nextHopNodeId': path[idx + 1] if idx + 1 < len(path) else path[-1]}
                        else:
                            response = {'path': path, 'nextHopNodeId': path[1] if len(path) > 1 else path[0]}
                    else:
                        response = {'error': 'No path found'}
                        print(f"❌ No path found for {key} with current: {current}")
                
                # ✅ CALCULATE AND LOG METRICS FOR ALL PATHS (cached or new)
                if path and len(path) > 1:
                    total_latency = 0
                    min_bandwidth = float('inf')
                    total_bandwidth = 0 
                    total_loss = 0
                    
                    print(f"\n{'─'*60}")
                    print(f"📈 Path Metrics Calculation:")
                    print(f"{'─'*60}")
                    
                    for i in range(len(path) - 1):
                        latency, loss_rate, bandwidth_mbps = env._calculate_link_metrics(
                            path[i], path[i + 1], packet
                        )
                        total_latency += latency
                        min_bandwidth = min(min_bandwidth, bandwidth_mbps)
                        total_bandwidth += bandwidth_mbps  # ← THÊM: Cộng dồn bandwidth
                        total_loss += loss_rate / (len(path) - 1)
                        
                        print(f"  Hop {i+1}: {path[i]:12s} → {path[i+1]:12s} | "
                              f"Latency: {latency:6.2f}ms | "
                              f"BW: {bandwidth_mbps:6.2f}Mbps | "
                              f"Loss: {loss_rate:.4f}")
                    
                    # Calculate average bandwidth
                    avg_bandwidth = total_bandwidth / (len(path) - 1)  # ← THÊM
                    
                    print(f"{'─'*60}")
                    print(f"📊 TOTAL PATH METRICS {'(CACHED)' if is_cached else '(NEW)'}:")
                    print(f"{'─'*60}")
                    print(f"  🛤️  Full Path      : {' → '.join(path)}")
                    print(f"  ⏱️  Total Latency  : {total_latency:.2f} ms")
                    print(f"  📶 Total Bandwidth : {total_bandwidth:.2f} Mbps")  # ← THÊM
                    print(f"  📊 Avg Bandwidth   : {avg_bandwidth:.2f} Mbps")    # ← THÊM
                    print(f"  📉 Min Bandwidth   : {min_bandwidth:.2f} Mbps")    # ← GIỮ LẠI
                    print(f"  📉 Avg Loss Rate   : {total_loss:.4f}")
                    print(f"  🔢 Hop Count       : {len(path) - 1}")
                    print(f"{'='*60}\n")
                    
                    # Add metrics to response
                    response['totalLatencyMs'] = str(total_latency)
                    response['totalBandwidthMbps'] = str(total_bandwidth)    # ← THÊM
                    response['avgBandwidthMbps'] = str(avg_bandwidth)        # ← THÊM
                    response['minBandwidthMbps'] = str(min_bandwidth)
                    response['lossRate'] = str(total_loss)
            
            client_socket.send(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"❌ Error handling client: {e}")
            import traceback
            traceback.print_exc()
            response = {'error': str(e)}
            client_socket.send(json.dumps(response).encode('utf-8'))
        finally:
            client_socket.close()
    
    def run(self):
        def train():
            while True:
                self.agent.replay()
                time.sleep(10)
        
        threading.Thread(target=train, daemon=True).start()
        
        while True:
            try:
                client_socket, addr = self.server_socket.accept()
                print(f"Connection from {addr}")
                threading.Thread(target=self.handle_client, args=(client_socket,)).start()
            except KeyboardInterrupt:
                self.save_checkpoint()
                self.server_socket.close()
                break

if __name__ == "__main__":
    server = RLServer()
    server.run()