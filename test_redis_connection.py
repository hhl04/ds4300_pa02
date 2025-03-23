import redis
import sys

try:
    r = redis.Redis(host="localhost", port=6379)  # Use your configured port
    if r.ping():
        print("Successfully connected to Redis!")
        # Try a simple operation
        r.set("test_key", "test value")
        print(f"Read test value: {r.get('test_key').decode('utf-8')}")
    else:
        print("Failed to connect to Redis, unable to ping")
except Exception as e:
    print(f"Connection error: {str(e)}")
    sys.exit(1)