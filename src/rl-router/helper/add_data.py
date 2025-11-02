import json
import pymongo
import logging
from datetime import datetime
from bson import json_util  # ✅ Dùng để hiểu được $oid, $date, $numberDouble, v.v.

# ----------------- Logger -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("add_data")

# ----------------- Config -----------------
DB_URI = "mongodb://user:password123@localhost:27017/?authSource=admin"
DB_NAME = "sagsin_network"
COLLECTION_NAME = "network_nodes"
JSON_FILE = "/Users/anhnon/PBL4/src/rl-router/helper/network_nodes.json"

# ----------------- Connect MongoDB -----------------
try:
    client = pymongo.MongoClient(DB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    logger.info(f"✅ Connected to MongoDB Atlas: {DB_NAME}.{COLLECTION_NAME}")
except Exception as e:
    logger.error(f"❌ Failed to connect MongoDB: {e}")
    exit(1)

# ----------------- Read & Parse JSON -----------------
try:
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        # ✅ json_util.loads sẽ tự chuyển $oid, $numberDouble, $date,... về đúng kiểu MongoDB
        nodes = json_util.loads(f.read())
    logger.info(f"📄 Loaded {len(nodes)} nodes from {JSON_FILE}")
except Exception as e:
    logger.error(f"❌ Failed to read JSON: {e}")
    exit(1)

# ----------------- Clean and Normalize -----------------
for node in nodes:
    # ⚙️ Bỏ _id cũ để MongoDB tự tạo mới (tránh lỗi $oid)
    node.pop("_id", None)

    # ⏰ Convert lastUpdated string → datetime (nếu chưa tự parse)
    if "lastUpdated" in node and isinstance(node["lastUpdated"], str):
        try:
            node["lastUpdated"] = datetime.fromisoformat(node["lastUpdated"].replace("Z", "+00:00"))
        except Exception:
            logger.warning(f"⚠️ Could not parse lastUpdated for nodeId={node.get('nodeId')}")

# ----------------- Insert to MongoDB -----------------
try:
    result = collection.delete_many({})
    logger.info(f"🧹 Cleared {result.deleted_count} old records.")

    collection.insert_many(nodes)
    logger.info(f"✅ Successfully inserted {len(nodes)} nodes into '{DB_NAME}.{COLLECTION_NAME}'")
except Exception as e:
    logger.error(f"❌ Insert failed: {e}")
