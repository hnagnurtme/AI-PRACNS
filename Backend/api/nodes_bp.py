"""
Nodes API Blueprint
Provides endpoints for managing network nodes (satellites, ground stations)
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from bson import ObjectId
from models.database import db
import logging

logger = logging.getLogger(__name__)

nodes_bp = Blueprint('nodes', __name__, url_prefix='/api/v1/nodes')

@nodes_bp.route('', methods=['GET'])
def get_all_nodes():
    """Get all nodes"""
    try:
        collection = db.get_collection('nodes')
        nodes = list(collection.find({}, {'_id': 0}))
        
        return jsonify(nodes), 200
        
    except Exception as e:
        logger.error(f"Error fetching nodes: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@nodes_bp.route('/<node_id>', methods=['GET'])
def get_node_by_id(node_id: str):
    """Get node by ID"""
    try:
        collection = db.get_collection('nodes')
        node = collection.find_one({'nodeId': node_id}, {'_id': 0})
        
        if not node:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Node with ID {node_id} not found'
            }), 404
        
        return jsonify(node), 200
        
    except Exception as e:
        logger.error(f"Error fetching node: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@nodes_bp.route('/<node_id>', methods=['PATCH'])
def update_node_status(node_id: str):
    """Update node status (partial update)"""
    try:
        data = request.get_json() or {}
        collection = db.get_collection('nodes')
        
        # Check if node exists
        node = collection.find_one({'nodeId': node_id})
        if not node:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Node with ID {node_id} not found'
            }), 404
        
        # Build update document
        update_data = {}
        allowed_fields = [
            'isOperational',
            'batteryChargePercent',
            'nodeProcessingDelayMs',
            'packetLossRate',
            'resourceUtilization',
            'currentPacketCount',
            'weather'
        ]
        
        for field in allowed_fields:
            if field in data:
                update_data[field] = data[field]
        
        # Add lastUpdated timestamp
        update_data['lastUpdated'] = datetime.now().isoformat()
        
        # Update node
        collection.update_one(
            {'nodeId': node_id},
            {'$set': update_data}
        )
        
        # Get updated node
        updated_node = collection.find_one({'nodeId': node_id}, {'_id': 0})
        
        logger.info(f"Updated node {node_id}: {list(update_data.keys())}")
        return jsonify(updated_node), 200
        
    except Exception as e:
        logger.error(f"Error updating node: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@nodes_bp.route('/<node_id>', methods=['PUT'])
def update_node(node_id: str):
    """Update entire node (full update)"""
    try:
        data = request.get_json() or {}
        collection = db.get_collection('nodes')
        
        # Check if node exists
        node = collection.find_one({'nodeId': node_id})
        if not node:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Node with ID {node_id} not found'
            }), 404
        
        # Preserve nodeId and id
        data['nodeId'] = node_id
        data['id'] = node.get('id', node_id)
        data['lastUpdated'] = datetime.now().isoformat()
        
        # Replace node
        collection.replace_one(
            {'nodeId': node_id},
            data
        )
        
        # Get updated node
        updated_node = collection.find_one({'nodeId': node_id}, {'_id': 0})
        
        logger.info(f"Replaced node {node_id}")
        return jsonify(updated_node), 200
        
    except Exception as e:
        logger.error(f"Error updating node: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@nodes_bp.route('/<node_id>', methods=['DELETE'])
def delete_node(node_id: str):
    """Delete node"""
    try:
        collection = db.get_collection('nodes')
        result = collection.delete_one({'nodeId': node_id})
        
        if result.deleted_count == 0:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Node with ID {node_id} not found'
            }), 404
        
        logger.info(f"Deleted node {node_id}")
        return jsonify({'message': 'Node deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Error deleting node: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@nodes_bp.route('', methods=['POST'])
def create_node():
    """Create a new node"""
    try:
        data = request.get_json() or {}
        collection = db.get_collection('nodes')
        
        # Validate required fields
        required_fields = ['nodeId', 'nodeName', 'nodeType', 'position']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 400,
                    'error': 'Bad request',
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Check if nodeId already exists
        existing = collection.find_one({'nodeId': data['nodeId']})
        if existing:
            return jsonify({
                'status': 409,
                'error': 'Conflict',
                'message': f'Node with ID {data["nodeId"]} already exists'
            }), 409
        
        # Set defaults
        data.setdefault('isOperational', True)
        data.setdefault('healthy', True)
        data.setdefault('batteryChargePercent', 100.0)
        data.setdefault('nodeProcessingDelayMs', 10.0)
        data.setdefault('packetLossRate', 0.0)
        data.setdefault('resourceUtilization', 0.0)
        data.setdefault('packetBufferCapacity', 1000)
        data.setdefault('currentPacketCount', 0)
        data.setdefault('weather', 'CLEAR')
        data.setdefault('lastUpdated', datetime.now().isoformat())
        
        # Insert node
        result = collection.insert_one(data)
        data['_id'] = str(result.inserted_id)
        
        # Remove _id from response
        node = collection.find_one({'nodeId': data['nodeId']}, {'_id': 0})
        
        logger.info(f"Created node {data['nodeId']}")
        return jsonify(node), 201
        
    except Exception as e:
        logger.error(f"Error creating node: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@nodes_bp.route('/<node_id>/health', methods=['GET'])
def get_node_health(node_id: str):
    """Get node health status"""
    try:
        collection = db.get_collection('nodes')
        node = collection.find_one({'nodeId': node_id}, {'_id': 0})
        
        if not node:
            return jsonify({
                'status': 404,
                'error': 'Not found',
                'message': f'Node with ID {node_id} not found'
            }), 404
        
        health_info = {
            'nodeId': node_id,
            'healthy': node.get('healthy', False),
            'isOperational': node.get('isOperational', False),
            'batteryChargePercent': node.get('batteryChargePercent', 0),
            'resourceUtilization': node.get('resourceUtilization', 0),
            'packetLossRate': node.get('packetLossRate', 0),
            'lastUpdated': node.get('lastUpdated')
        }
        
        return jsonify(health_info), 200
        
    except Exception as e:
        logger.error(f"Error fetching node health: {str(e)}")
        return jsonify({
            'status': 500,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

