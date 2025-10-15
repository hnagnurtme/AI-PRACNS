from pymongo import MongoClient
import os
MONGO_URI = os.getenv("MONGO_URL", "mongodb://user:password123@localhost:27017/")
DB_NAME = "sagsin_network"

def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

def get_collection(name="nodes"):
    db = get_db()
    return db[name]
