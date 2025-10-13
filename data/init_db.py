from pymongo import ASCENDING
from db_config import get_collection

def create_indexes():
    nodes = get_collection("nodes")
    nodes.create_index([("nodeId", ASCENDING)], unique=True)
    nodes.create_index([("type", ASCENDING)])
    nodes.create_index([("status.active", ASCENDING)])
    nodes.create_index([("position.latitude", ASCENDING), ("position.longitude", ASCENDING)])
    nodes.create_index([("lastUpdated", ASCENDING)])

    print("Indexes created successfully.")

if __name__ == "__main__":
    create_indexes()
