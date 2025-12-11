#!/usr/bin/env python3
"""
Script to initialize database with sample data
"""
from pymongo import MongoClient
import os

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://admin:password@mongodb:27017/aiprancs?authSource=admin')
DB_NAME = os.getenv('DB_NAME', 'aiprancs')

try:
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    
    # Create indexes
    messages_collection = db['messages']
    messages_collection.create_index('timestamp')
    
    print("✅ Database initialized successfully")
    client.close()
except Exception as e:
    print(f"⚠️  Database initialization warning: {e}")

