"""MongoDB connection and database initialization."""

import os
import time
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi

# Load environment variables from Vercel
load_dotenv()

# Load MongoDB connection string from env
MONGO_CONNECTION_STRING = os.getenv("MONGO_DB_CONNECTION_STRING")

if not MONGO_CONNECTION_STRING:
    raise ValueError("MONGO_DB_CONNECTION_STRING not found in environment variables")

# Use certifi's CA bundle for consistent TLS verification
client = MongoClient(
    MONGO_CONNECTION_STRING,
    serverSelectionTimeoutMS=10000,
    connectTimeoutMS=10000,
    socketTimeoutMS=20000,
    tlsCAFile=certifi.where(),
    connect=False,  # Defer connection for serverless
)

# Get database and collection references (lazy — no connection yet)
db = client["hackx_db"]
users_collection = db["users"]


def ensure_indexes():
    """Create required indexes. Call on app startup. Retries on transient SSL/network errors."""
    last_error = None
    for attempt in range(1, 4):
        try:
            client.admin.command("ping")
            users_collection.create_index("email", unique=True)
            users_collection.create_index("created_at")
            print("✓ MongoDB connected and indexes ensured")
            return
        except Exception as e:
            last_error = e
            if attempt < 3:
                time.sleep(2)
    print(f"⚠ MongoDB connection/index failed after 3 attempts: {last_error}")
