import time
import os
from datetime import datetime
from pymongo import MongoClient

# --- Configuration ---
# Use environment variables with fallback defaults
LOCAL_MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://user:password123@localhost:27018/")
CLOUD_MONGO_URI = os.getenv("CLOUD_MONGO_URI", "mongodb+srv://admin:SMILEisme0106@mongo1.ragz4ka.mongodb.net/network?retryWrites=true&w=majority&tls=true&appName=MONGO1&tlsAllowInvalidCertificates=true")
DB_NAME = os.getenv("DB_NAME", "sagsin_network")
SYNC_INTERVAL_SECONDS = int(os.getenv("SYNC_INTERVAL_SECONDS", "15"))  # Sync every 15 seconds

def get_mongo_client(uri):
    """Creates a MongoClient from a URI."""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster') # Check connection
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB at {uri}: {e}")
        return None

def synchronize_databases():
    """
    Synchronizes the local 'network_nodes' collection to the cloud 'network_nodes' collection.
    This implementation performs a full overwrite of the cloud collection with the local data.
    """
    print(f"[{datetime.now()}] Starting database synchronization...")

    local_client = get_mongo_client(LOCAL_MONGO_URI)
    cloud_client = get_mongo_client(CLOUD_MONGO_URI)

    if not local_client:
        print("Failed to connect to local MongoDB. Skipping sync.")
        return
    if not cloud_client:
        print("Failed to connect to cloud MongoDB. Skipping sync.")
        local_client.close()
        return

    try:
        local_db = local_client[DB_NAME]
        cloud_db = cloud_client[DB_NAME]

        local_nodes_collection = local_db["network_nodes"]
        cloud_nodes_collection = cloud_db["network_nodes"]

        # 1. Fetch all nodes from the local database
        local_nodes = list(local_nodes_collection.find({}))
        print(f"  Fetched {len(local_nodes)} nodes from local database.")

        # 2. Overwrite the cloud database's network_nodes collection
        print("  Deleting all existing nodes from cloud database...")
        delete_result = cloud_nodes_collection.delete_many({})
        print(f"  Deleted {delete_result.deleted_count} nodes from cloud.")

        if local_nodes:
            print(f"  Inserting {len(local_nodes)} nodes into cloud database...")
            cloud_nodes_collection.insert_many(local_nodes)
            print("  Nodes inserted into cloud database.")
        else:
            print("  No nodes to insert from local database.")

        print(f"[{datetime.now()}] Database synchronization complete.")

    except Exception as e:
        print(f"Error during synchronization: {e}")
    finally:
        if local_client:
            local_client.close()
        if cloud_client:
            cloud_client.close()

if __name__ == "__main__":
    print("Starting database synchronization scheduler...")
    try:
        while True:
            synchronize_databases()
            time.sleep(SYNC_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nSynchronization scheduler stopped by user.")
