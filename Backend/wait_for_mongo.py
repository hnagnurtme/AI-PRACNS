#!/usr/bin/env python3
"""
Script to wait for MongoDB to be ready
"""
import time
import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os

MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://admin:password@mongodb:27017/aiprancs?authSource=admin')
MAX_RETRIES = 30
RETRY_INTERVAL = 2

def wait_for_mongo():
    """Wait for MongoDB to be available"""
    print("Waiting for MongoDB to be ready...")
    
    for i in range(MAX_RETRIES):
        try:
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
            client.admin.command('ping')
            print("✅ MongoDB is ready!")
            client.close()
            return True
        except ConnectionFailure:
            print(f"⏳ Attempt {i+1}/{MAX_RETRIES}: MongoDB not ready yet, waiting...")
            time.sleep(RETRY_INTERVAL)
        except Exception as e:
            print(f"⚠️  Error: {e}")
            time.sleep(RETRY_INTERVAL)
    
    print("❌ Failed to connect to MongoDB after maximum retries")
    return False

if __name__ == '__main__':
    if not wait_for_mongo():
        sys.exit(1)

