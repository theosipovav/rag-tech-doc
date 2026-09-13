# RAG PDF Chat System

A production-ready Retrieval-Augmented Generation (RAG) system for chatting with PDF documents. Features offline indexing and online query pipelines with state-of-the-art models.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OFFLINE PIPELINE (Indexing)                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │ PDF Files│───▶│  Parser  │───▶│ Chunking │───▶│  Embeddings  │──┼──▶ Qdrant DB
│  │          │    │(MinerU/  │    │(Sentence │    │  (BGE-M3)    │  │
│  │          │    │PyMuPDF)  │    │Splitter) │    │              │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     ONLINE PIPELINE (Query)                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │  User    │───▶│ Embedding│───▶│  Vector  │───▶│   Reranker   │  │
│  │ Question │    │ (BGE-M3) │    │  Search  │    │(BGE v2)      │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────┘  │
│                                                        │            │
│  ┌──────────┐    ┌──────────┐                          ▼            │
│  │ Response │◀───│   LLM    │◀─────────────────┌──────────────┐    │
│  │ + Sources│    │(vLLM/    │                  │  Context     │    │
│  │          │    │ Ollama)  │                  │  Assembly    │    │
│  └──────────┘    └──────────┘                  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## 📋 Features

- **Dual Pipeline Architecture**: Separate offline indexing and online query pipelines
- **Advanced PDF Parsing**: Support for complex PDFs with tables, formulas, and OCR
- **Semantic Chunking**: Intelligent text splitting using LlamaIndex SentenceSplitter
- **State-of-the-Art Embeddings**: BGE-M3 model for dense vector representations
- **Vector Search**: High-performance similarity search with Qdrant
- **Result Reranking**: BGE Reranker v2 for improved relevance
- **LLM Integration**: Compatible with vLLM and Ollama for fast inference
- **Modern UI**: Streamlit-based frontend with chat interface
- **REST API**: FastAPI backend with OpenAPI documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional, for containerized deployment)
- NVIDIA GPU with CUDA support (recommended for best performance)

### Option 1: Local Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag-pdf-chat
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start Qdrant**
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Start the backend**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

6. **Start the frontend** (in a new terminal)
```bash
streamlit run frontend/app.py
```

7. **Access the application**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Option 2: Docker Compose

1. **Start all services**
```bash
docker-compose up -d
```

2. **For GPU acceleration**
```bash
docker-compose --profile gpu up -d
```

3. **Access the application**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8001
- Qdrant Dashboard: http://localhost:6333/dashboard

## 📁 Project Structure

```
rag-pdf-chat/
├── app/                        # Backend application
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── schemas.py             # Pydantic models
│   ├── embeddings.py          # BGE-M3 embedding service
│   ├── vector_db.py           # Qdrant vector database
│   ├── reranker.py            # BGE reranker service
│   ├── llm.py                 # LLM inference service
│   ├── pdf_parser.py          # PDF parsing service
│   ├── chunking.py            # Text chunking service
│   ├── rag_pipeline.py        # RAG orchestration
│   └── main.py                # FastAPI application
├── frontend/                   # Streamlit frontend
│   ├── app.py                 # Main Streamlit app
│   └── requirements.txt       # Frontend dependencies
├── data/                       # Data directory
│   └── uploads/               # Uploaded PDF files
├── logs/                       # Application logs
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile.backend         # Backend Docker image
├── Dockerfile.frontend        # Frontend Docker image
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | localhost | Qdrant server host |
| `QDRANT_PORT` | 6333 | Qdrant server port |
| `EMBEDDING_MODEL_NAME` | BAAI/bge-m3 | Embedding model |
| `RERANKER_MODEL_NAME` | BAAI/bge-reranker-v2-m3 | Reranker model |
| `LLM_API_BASE` | http://localhost:8000/v1 | LLM API endpoint |
| `LLM_MODEL_NAME` | meta-llama/Llama-3-8B-Instruct | LLM model |
| `CHUNK_SIZE` | 512 | Text chunk size |
| `CHUNK_OVERLAP` | 50 | Chunk overlap |

## 📖 API Endpoints

### Documents
- `POST /upload` - Upload and index a PDF document
- `POST /upload/batch` - Upload multiple PDFs

### Chat
- `POST /chat` - Chat with indexed documents

### Search
- `POST /search` - Semantic search without LLM generation

### System
- `GET /health` - Health check
- `GET /stats` - System statistics
- `DELETE /reset` - Reset vector database

## 🎯 Usage Examples

### Upload a Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### Chat with Documents
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the key findings?",
    "top_k": 5,
    "include_sources": true
  }'
```

### Semantic Search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "top_k": 10
  }'
```

## 🧠 Models

The system uses the following models by default:

- **Embeddings**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) - Multi-lingual, multi-granularity embeddings
- **Reranker**: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) - Cross-encoder for reranking
- **LLM**: Compatible with any OpenAI-compatible API (vLLM, Ollama, etc.)

## ⚡ Performance Tips

1. **GPU Acceleration**: Use NVIDIA GPU for embedding, reranking, and LLM inference
2. **Batch Processing**: Index multiple documents in batch for efficiency
3. **Chunk Size**: Adjust chunk size based on your documents (512-1024 tokens recommended)
4. **Top-K**: Tune retrieval parameters for your use case
5. **Quantization**: Use quantized LLM models for reduced VRAM usage

## 🔒 Security Considerations

- Change default CORS settings for production
- Implement authentication for production use
- Secure API endpoints with rate limiting
- Use HTTPS in production environments

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.