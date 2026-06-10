# 🅿️ InstaParkAI - Smart Parking RAG Chatbot

An industry-leading, AI-powered Smart Parking Knowledge Assistant. This application utilizes a Retrieval-Augmented Generation (RAG) pipeline to ingest parking manuals, pricing sheets, contracts, and FAQs, providing instant, context-aware answers about smart parking technologies (ANPR, RFID, IoT sensors), operations, and contracts.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Make sure you have **Python 3.10+** installed on your system. 

### 2. Environment Configuration
The application loads its settings from a `.env` file in the root directory. An `.env` file has already been configured with the appropriate API keys:
- **Groq API Key**: Configured for LLM inference using LLaMA models.
- **Gemini Config**: Configured for Gemini models.
- **Embeddings & Reranker**: Configured to run locally using SentenceTransformers (on CPU).

### 3. Install Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Generate Sample Parking Knowledge Base
Before querying, you need source documents. We have provided a script that generates sample parking PDFs (company overview, FAQs, pricing models, and maintenance contracts):
```bash
python scripts/generate_parking_pdfs.py
```
This will create sample PDF documents under the `data/uploads/` directory.

### 5. Ingest the Documents (Indexing)
To load these documents into the vector store (FAISS) and keyword search index (BM25), run the ingestion commands:
```bash
python -m app.main --ingest data/uploads/system_overview.pdf
python -m app.main --ingest data/uploads/pricing_and_amc.pdf
python -m app.main --ingest data/uploads/faq.pdf
```
*(Alternatively, you can skip this step as the Evaluation runner will auto-ingest these files if the index is empty).*

---

## 💻 How to Run the Application

You can interact with InstaParkAI in the following ways:

### A. Command-Line Interface (CLI)
You can run individual queries directly from your terminal:
```bash
python -m app.main --query "What is ANPR technology?"
```
```bash
python -m app.main --query "What is the price of the SaaS subscription?"
```

### B. Run the Evaluation Suite
Validate the performance, intent classification, hallucination guard, and retrieval accuracy across a standard dataset:
```bash
python scripts/run_evaluation.py
```
- This runs positive, multilingual, and trap queries.
- Generates a detailed JSON evaluation report saved under `data/evaluation/`.

---

## 🧪 Running Tests
Verify the pipeline integrity and unit tests using pytest:
```bash
pytest
```

---

## 🔌 REST API — Primary Interface (Recommended)

The project includes a FastAPI backend (`app/api.py`) that exposes the RAG pipeline as HTTP endpoints. This allows any frontend — React, Vue, Angular, mobile apps — to consume the chatbot service.

### Starting the API Server
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Send a question, receive an answer with citations |
| `POST` | `/api/ingest` | Upload and index a PDF document |
| `GET`  | `/api/health` | System health check (indexed chunks, LLM model) |
| `POST` | `/api/sessions/{session_id}/clear` | Clear chat history for a session |
| `POST` | `/api/cache/clear` | Flush the semantic cache |

### Example: Query (cURL)
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is ANPR technology?", "session_id": "user-123"}'
```

### Example: React Integration
```jsx
const response = await fetch('http://localhost:8000/api/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: userInput, session_id: sessionId }),
});
const data = await response.json();
// data.answer, data.citations, data.latency_ms, data.session_id
```

### Session Management
Each `session_id` gets its own conversation memory while sharing the same knowledge base, vector index, and semantic cache. If no `session_id` is provided, the server creates one automatically and returns it in the response.

---

## 🐳 Docker Deployment

The entire application is containerized using **Docker Compose** with three services: Redis (caching), FastAPI Backend, and Nginx Frontend.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.
- A `.env` file in the project root with your API keys (see [Environment Configuration](#2-environment-configuration)).

### Quick Start (One Command)
```bash
docker-compose up --build -d
```
This builds and starts all three services in detached mode. The first build may take a few minutes as it downloads ML models.

### Services Architecture

| Service | Image / Build | Container Name | Port Mapping | Description |
|---------|---------------|----------------|--------------|-------------|
| **Redis** | `redis:alpine` | `instapark-redis` | `6379:6379` | In-memory cache for semantic query caching |
| **Backend** | `Dockerfile.backend` | `instapark-backend` | `8000:8000` | FastAPI RAG pipeline server (Uvicorn) |
| **Frontend** | `Dockerfile.frontend` | `instapark-frontend` | `5500:80` | Static web UI served via Nginx |

### Service URLs (after `docker-compose up`)

| Service | URL |
|---------|-----|
| Frontend UI | [http://localhost:5500](http://localhost:5500) |
| Backend API | [http://localhost:8000](http://localhost:8000) |
| Swagger Docs | [http://localhost:8000/docs](http://localhost:8000/docs) |
| ReDoc | [http://localhost:8000/redoc](http://localhost:8000/redoc) |
| Health Check | [http://localhost:8000/api/health](http://localhost:8000/api/health) |

### Docker Compose Configuration (`docker-compose.yml`)
```yaml
services:
  redis:
    image: redis:alpine
    container_name: instapark-redis
    ports:
      - "6379:6379"
    restart: always

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: instapark-backend
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - GPTCACHE_ENABLED=true
    volumes:
      - ./data:/workspace/data
    depends_on:
      - redis
    restart: always

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    container_name: instapark-frontend
    ports:
      - "5500:80"
    restart: always
```

### Backend Dockerfile (`Dockerfile.backend`)
- **Base image**: `python:3.11-slim`
- Installs PyTorch (CPU-only) to keep image lightweight
- Pre-downloads SentenceTransformer and CrossEncoder models at build time (no download on startup)
- Runs `uvicorn app.api:app` on port `8000`

### Frontend Dockerfile (`Dockerfile.frontend`)
- **Base image**: `nginx:alpine`
- Copies the `web/` static assets into Nginx's default HTML directory
- Serves the UI on port `80` (mapped to `5500` on host)

### Environment Variables
The backend container reads from your `.env` file plus these Docker-specific overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `redis` | Redis service hostname (Docker internal DNS) |
| `REDIS_PORT` | `6379` | Redis port |
| `GPTCACHE_ENABLED` | `true` | Enable/disable semantic caching |

### Common Docker Commands
```bash
# Build and start all services
docker-compose up --build -d

# View logs for all services
docker-compose logs -f

# View logs for a specific service
docker-compose logs -f backend

# Stop all services
docker-compose down

# Rebuild a specific service
docker-compose up --build -d backend

# Check running containers
docker ps

# Access backend container shell
docker exec -it instapark-backend bash

# Check backend health
curl http://localhost:8000/api/health
```

### Volumes
The `./data` directory is mounted into the backend container at `/workspace/data`, ensuring:
- Uploaded PDFs persist across container restarts
- FAISS and BM25 indexes are preserved
- Evaluation reports are accessible from the host

---

## ☁️ Cloud Deployment (Render — All-in-One)

Deploy the entire stack (Backend + Frontend + Redis) on **[Render](https://render.com)** from a single platform — no separate hosting needed.

### How It Works
- The FastAPI backend **also serves the frontend** static files, so only **one web service** is needed
- Redis is provisioned as a managed service on Render
- Auto-deploys from your GitHub repo on every push

### One-Click Deploy
1. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**
2. Connect your GitHub repo: `NACNAVEEN/Customer_query_RAG_BOT`
3. Render auto-detects the `render.yaml` and creates both services
4. Set your **API keys** in the Render dashboard:
   - `GROQ_API_KEY` — your Groq API key
   - `GOOGLE_API_KEY` — your Google/Gemini API key
5. Click **Apply** — Render builds and deploys everything

### What Gets Deployed

| Service | Type | Plan | Description |
|---------|------|------|-------------|
| `instapark-ai` | Web Service (Docker) | Free | FastAPI backend + static frontend |
| `instapark-redis` | Redis | Free | Managed Redis cache |

### After Deployment
Your app will be live at:
```
https://instapark-ai.onrender.com        → Frontend UI
https://instapark-ai.onrender.com/api/   → API endpoints
https://instapark-ai.onrender.com/docs   → Swagger documentation
```

> **Note**: Free-tier Render services spin down after 15 minutes of inactivity. The first request after idle may take ~30 seconds to cold-start.

### Render Configuration (`render.yaml`)
The `render.yaml` Blueprint defines all services. Render reads this file automatically from your repo.

---

## 🏗️ Project Structure
- `app/` - Core application codebase.
  - `api.py` - **FastAPI REST API backend** (also serves frontend in production).
  - `config/` - Configuration settings.
  - `pipeline/` - RAG Graph and orchestration.
  - `validators/` - Hallucination guards and intent validators.
- `web/` - Static frontend (HTML/CSS/JS).
- `data/` - Locally stored data (uploads, indexes, evaluation reports).
- `scripts/` - Utility scripts for generating PDFs and executing evaluations.
- `docker-compose.yml` - Multi-service Docker orchestration (local development).
- `render.yaml` - Render Blueprint for cloud deployment.
- `Dockerfile.backend` - Backend container build configuration.
- `Dockerfile.frontend` - Frontend container build configuration.

