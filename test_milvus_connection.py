from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

def test_milvus_connection():
    try:
        # Try to connect to Milvus
        print("Connecting to Milvus...")
        connections.connect("default", host="localhost", port="19530")
        print("✅ Milvus connection successful!")
        
        # List all collections
        print("\nCurrent collections:")
        collections = utility.list_collections()
        for collection in collections:
            print(f"- {collection}")
        
        # Disconnect
        connections.disconnect("default")
        print("\n✅ Connection test completed")
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")

if __name__ == "__main__":
    test_milvus_connection() 