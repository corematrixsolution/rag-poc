# Upgrade Chocolatey itself:
choco upgrade chocolatey

# https://www.youtube.com/watch?v=DHjk5Un0Kxs
# https://github.com/neelam-yadav/rag-poc

# Open power shell
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-community langchain-qdrant langchain-ollama langchain-experimental qdrant-client sentence-transformers fastapi uvicorn[standard] pydantic requests beautifulsoup4 python-dotenv tiktoken

# Open new power shell
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage --name qdrant -d qdrant/qdrant:latest
# Verify Qdrant Vector DB is running
http://localhost:6333/dashboard

# Install Ollama (if you don't have it alrady) or use choco
https://ollama.com/download/windows
# OR
choco install ollama


# Pull ministral-3:3b
ollama pull ministral-3:3b

#
python -m pipeline.build_index

#
uvicorn app.main:app --reload --port 8000


pip install -U langchain langchain-core langchain-community
pip install -U langchain-ollama langchain-qdrant qdrant-client