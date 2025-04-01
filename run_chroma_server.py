# run_chroma_server.py
import uvicorn
from chromadb.app import app  # ✅ this is where the FastAPI app lives in 0.6+

uvicorn.run(app, host="0.0.0.0", port=8000)
