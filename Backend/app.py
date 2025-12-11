"""
Main Flask application for SAGIN RL Backend
"""
from flask import Flask, jsonify
from flask_cors import CORS  # pyright: ignore[reportMissingModuleSource]
from flask_socketio import SocketIO  # pyright: ignore[reportMissingModuleSource]
from datetime import datetime
import logging
from config import Config
from models.database import db
from api import terminals_bp, nodes_bp, routing_bp, topology_bp, simulation_bp, batch_bp

# Import other blueprints if they exist
try:
    from api import networks_bp, training_bp
except ImportError:
    networks_bp = None
    training_bp = None
    allocation_bp = None

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Initialize SocketIO for real-time updates
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    logger=True,
    engineio_logger=True
)

# Store socketio in app extensions for routing_bp to access
app.extensions['socketio'] = socketio

# Configure CORS to allow credentials for WebSocket and all origins
CORS(app, 
     resources={
         r"/api/*": {
             "origins": "*", 
             "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"], 
             "allow_headers": ["Content-Type", "Authorization"],
             "supports_credentials": True
         },
         r"/ws/*": {
             "origins": "*",
             "methods": ["GET", "POST", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization"],
             "supports_credentials": True
         },
     },
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     expose_headers=["Content-Type"])


# Note: CORS is already handled by Flask-CORS above
# Don't add duplicate headers here to avoid "multiple values" error

# Register blueprints
if networks_bp:
    app.register_blueprint(networks_bp)
if training_bp:
    app.register_blueprint(training_bp)
# Removed allocation_bp as it is not a valid import
app.register_blueprint(terminals_bp)
app.register_blueprint(nodes_bp)
app.register_blueprint(routing_bp)
app.register_blueprint(topology_bp)
app.register_blueprint(simulation_bp)
app.register_blueprint(batch_bp)

# Auto-start batch streaming on app initialization
def init_batch_streaming():
    """Initialize batch streaming on app start"""
    import threading
    import time
    from api.batch_bp import _batch_generation_active, _batch_generation_thread
    from api.batch_bp import calculate_path_dijkstra, calculate_path_rl, create_packet_from_path
    from models.database import db
    import random
    
    def generate_and_queue():
        """Background thread to generate batches"""
        while True:
            try:
                # Check if already active
                if _batch_generation_active:
                    time.sleep(5)
                    continue
                
                # Generate batch
                terminals_collection = db.get_collection('terminals')
                terminals = list(terminals_collection.find({}, {'_id': 0}))
                
                if len(terminals) < 2:
                    time.sleep(5)
                    continue
                
                nodes_collection = db.get_collection('nodes')
                nodes = list(nodes_collection.find({'isOperational': True}, {'_id': 0}))
                
                if len(nodes) == 0:
                    time.sleep(5)
                    continue
                
                batch_id = f"BATCH-{int(datetime.now().timestamp() * 1000)}"
                packets = []
                pair_count = 10
                
                for i in range(pair_count):
                    source_terminal = random.choice(terminals)
                    dest_terminal = random.choice(terminals)
                    while dest_terminal['terminalId'] == source_terminal['terminalId']:
                        dest_terminal = random.choice(terminals)
                    
                    service_qos = {
                        'maxLatencyMs': random.randint(50, 200),
                        'minBandwidthMbps': random.uniform(0.5, 5.0),
                        'maxLossRate': random.uniform(0.001, 0.05),
                        'priority': random.randint(1, 10),
                        'serviceType': random.choice(['TEXT_MESSAGE', 'IMAGE_TRANSFER', 'VIDEO_STREAM'])
                    }
                    
                    path_dijkstra = calculate_path_dijkstra(source_terminal, dest_terminal, nodes)
                    path_rl = calculate_path_rl(source_terminal, dest_terminal, nodes, service_qos=service_qos)
                    
                    dijkstra_packet = create_packet_from_path(path_dijkstra, source_terminal, dest_terminal, 'dijkstra', service_qos)
                    rl_packet = create_packet_from_path(path_rl, source_terminal, dest_terminal, 'rl', service_qos)
                    
                    packets.append({
                        'dijkstraPacket': dijkstra_packet,
                        'rlPacket': rl_packet
                    })
                
                batch = {
                    'batchId': batch_id,
                    'totalPairPackets': pair_count,
                    'packets': packets,
                    'scenario': 'NORMAL'  # Default scenario for auto-generated batches
                }
                
                from api.batch_bp import _batch_message_queue
                _batch_message_queue.append(batch)
                logging.info(f"Auto-generated batch {batch_id}")
                
            except Exception as e:
                logging.error(f"Error in auto batch generation: {str(e)}")
            
            time.sleep(5)  # Generate every 5 seconds
    
    def delayed_start():
        time.sleep(3)  # Wait for app to be ready
        thread = threading.Thread(target=generate_and_queue, daemon=True)
        thread.start()
        logging.info("Auto-started batch generation")
    
    thread = threading.Thread(target=delayed_start, daemon=True)
    thread.start()

# Start batch generation automatically
init_batch_streaming()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        db_status = "connected" if db.is_connected() else "disconnected"
        
        return jsonify({
            'status': 'healthy',
            'database': db_status,
            'timestamp': datetime.now().isoformat(),
            'service': 'SAGIN RL Backend'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'database': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api', methods=['GET'])
def api_info():
    """API information"""
    return jsonify({
        'name': 'SAGIN RL Backend API',
        'version': '1.0.0',
        'description': 'Resource allocation optimization for SAGIN using Reinforcement Learning',
        'endpoints': {
            'networks': '/api/networks',
            'training': '/api/training',
            'allocation': '/api/allocation',
            'nodes': '/api/v1/nodes',
            'terminals': '/api/v1/terminals',
            'routing': '/api/v1/routing',
            'health': '/api/health'
        }
    }), 200

@app.route('/ws/info', methods=['GET', 'OPTIONS'])
def sockjs_info():
    """SockJS info endpoint - required for SockJS client initialization"""
    import random
    import string
    
    # Generate random entropy (8 characters)
    entropy = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
    return jsonify({
        'websocket': True,  # We support WebSocket
        'origins': ['*:*'],  # Allow all origins
        'cookie_needed': False,
        'entropy': entropy
    }), 200

if __name__ == '__main__':
    port = Config.FLASK_PORT
    debug = Config.FLASK_ENV == 'development'
    # Use socketio.run instead of app.run for WebSocket support
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)
