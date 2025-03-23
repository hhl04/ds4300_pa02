import chromadb

def test_chroma_connection():
    try:
        # Try to connect to ChromaDB
        print("Connecting to ChromaDB...")
        client = chromadb.HttpClient(host="localhost", port="8000")
        
        # Test connection by getting server heartbeat
        client.heartbeat()
        print("✅ ChromaDB connection successful!")
        
        # List all collections
        print("\nCurrent collections:")
        collections = client.list_collections()
        for collection in collections:
            print(f"- {collection.name}")
        
        print("\n✅ Connection test completed")
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")

if __name__ == "__main__":
    test_chroma_connection() 