"""MongoDB connection and database initialization."""

import os
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# Load .env from project root (works regardless of CWD)
_root_env = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_root_env)

# Load MongoDB connection string from .env
MONGO_CONNECTION_STRING = os.getenv("MONGO_DB_CONNECTION_STRING")

if not MONGO_CONNECTION_STRING:
    raise ValueError("MONGO_DB_CONNECTION_STRING not found in .env file")

# Create MongoDB client
# tlsAllowInvalidCertificates bypasses LibreSSL cipher mismatch on macOS Python 3.9 (dev only)
client = MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=5000, tlsAllowInvalidCertificates=True)

# Get database reference
db = client["hackx_db"]

# Verify connection (non-fatal — transient Atlas SSL issues shouldn't kill the server)
try:
    client.admin.command("ping")
    print("✓ MongoDB connected successfully")
except Exception as e:
    print(f"⚠ MongoDB ping failed (may be transient): {e}")
    print("  Server continuing — DB operations will retry on next request.")

# Ensure indexes on users collection
users_collection = db["users"]
users_collection.create_index("email", unique=True)
users_collection.create_index("created_at")
