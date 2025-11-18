import os
from pymongo import MongoClient
from generate_asian_nodes import generate_asian_nodes_data

def get_mongo_client(uri):
    """Creates a MongoClient from a URI."""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB at {uri}: {e}")
        return None

def initialize_database(client, db_name="sagsin_network"):
    """
    Initializes the database with the 30-node Asian dataset.
    Deletes all existing nodes before inserting the new ones.
    """
    if not client:
        return

    db = client[db_name]
    nodes_collection = db["network_nodes"]

    # 1. Delete all existing nodes
    print(f"Deleting all existing nodes from '{db_name}.network_nodes'...")
    result = nodes_collection.delete_many({})
    print(f"Deleted {result.deleted_count} nodes.")

    # 2. Generate the new dataset
    print("Generating new 30-node Asian dataset...")
    new_nodes = generate_asian_nodes_data()

    # 3. Insert the new nodes
    print(f"Inserting {len(new_nodes)} new nodes...")
    nodes_collection.insert_many(new_nodes)
    print("Database initialization complete.")

if __name__ == "__main__":
    # --- Configuration ---
    # For local Docker MongoDB
    LOCAL_MONGO_URI = "mongodb://user:password123@localhost:27018/"

    # TODO: Replace with your cloud MongoDB URI when available
    CLOUD_MONGO_URI = os.environ.get("CLOUD_MONGO_URI", "mongodb+srv://admin:SMILEisme0106@mongo1.ragz4ka.mongodb.net/network?retryWrites=true&w=majority&tls=true&appName=MONGO1&tlsAllowInvalidCertificates=true")

    # --- Initialization ---
    print("--- Initializing Local Database ---")
    local_client = get_mongo_client(LOCAL_MONGO_URI)
    if local_client:
        initialize_database(local_client)
        local_client.close()
    else:
        print("Skipping local database initialization due to connection error.")

    if CLOUD_MONGO_URI:
        print("\n--- Initializing Cloud Database ---")
        cloud_client = get_mongo_client(CLOUD_MONGO_URI)
        if cloud_client:
            initialize_database(cloud_client)
            cloud_client.close()
        else:
            print("Skipping cloud database initialization due to connection error.")
    else:
        print("\n--- Skipping Cloud Database ---")
        print("CLOUD_MONGO_URI environment variable not set.")