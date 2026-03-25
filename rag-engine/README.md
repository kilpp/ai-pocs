# RAG Engine

A local Retrieval-Augmented Generation (RAG) pipeline CLI tool written in Rust. Ingest text/markdown documents into a vector index, then ask questions that are answered using retrieved context sent to the Claude API.

## Setup

### 1. Install the embedding model

Download the ONNX model files for `all-MiniLM-L6-v2`:

```bash
mkdir -p ~/.rag-engine/model
cd ~/.rag-engine/model

# Download model and tokenizer from HuggingFace
curl -L -o model.onnx "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
curl -L -o tokenizer.json "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=your-api-key-here
```

### 3. Build

```bash
cargo build --release
```

## Usage

### Ingest documents

```bash
# Ingest a single file
rag ingest --path ./docs/notes.md

# Ingest a directory (recursively finds .txt and .md files)
rag ingest --path ./docs/

# Custom chunk size and overlap
rag ingest --path ./docs/ --chunk-size 1024 --overlap 100
```

### Query

```bash
rag query --question "What are the main components of the system?"

# Retrieve more context chunks
rag query --question "How does authentication work?" --top-k 10
```

### List indexed documents

```bash
rag list
```

### Clear the index

```bash
rag clear
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-size` | 512 | Chunk size in characters |
| `--overlap` | 50 | Overlap between chunks in characters |
| `--top-k` | 5 | Number of context chunks to retrieve |
| `--model-dir` | `~/.rag-engine/model/` | Path to the ONNX model directory |
| `--index-path` | `~/.rag-engine/index.bin` | Path to the index file |

## Project Structure

```
src/
├── main.rs        CLI definition and orchestration
├── lib.rs         Module declarations
├── error.rs       Error types
├── document.rs    Document reading and text chunking
├── embedder.rs    ONNX model loading and inference
├── index.rs       HNSW vector index wrapper
├── store.rs       Index persistence (bincode)
└── llm.rs         Claude API client
```
