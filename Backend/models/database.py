"""
Database connection module
"""
from pymongo import MongoClient
from config import Config

class Database:
    """Database connection manager"""
    def __init__(self):
        self.client = None
        self.db = None
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(Config.MONGODB_URI)
            self.db = self.client[Config.DB_NAME]
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False
    
    def is_connected(self):
        """Check if database is connected"""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
            return False
        except:
            return False
    
    def get_collection(self, name):
        """Get a collection from the database"""
        if self.db is None:
            self.connect()
        return self.db[name] if self.db is not None else None
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()

# Global database instance
db = Database()
db.connect()

