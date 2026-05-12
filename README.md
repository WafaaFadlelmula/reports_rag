# Reports RAG Pipeline

A local Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions across a collection of PDF and Word reports. Built for synthesising technical project documentation into structured, professional answers.

## Overview

The pipeline indexes all your PDF and `.docx` files once, then lets you query them in a chat interface. Each question retrieves the most relevant excerpts and sends them to Claude (Anthropic) to generate a well-structured, cited answer. Answers can be saved as Markdown or Word documents.

```
reports/          ← put your PDF and .docx files here
chroma_db/        ← auto-created on first run (vector index)
outputs/          ← saved answers (.md and .docx)
rag_pipeline.py   ← main script
requirements.txt
```

## Features

- **Multi-format support** — indexes both PDF and `.docx` files
- **Local embeddings** — uses `all-MiniLM-L6-v2` via sentence-transformers (no external embedding API needed)
- **Persistent vector store** — ChromaDB stores the index locally; indexing only runs once
- **Two-stage retrieval** — semantic search (top 15 chunks) plus keyword-boosted retrieval for explicitly named projects
- **Claude-powered answers** — streams answers from `claude-opus-4-6` with source citations
- **Save outputs** — export any answer as `.md` or `.docx` with a timestamped filename

## Requirements

- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/)

## Installation

```bash
git clone https://github.com/your-username/reports-rag.git
cd reports-rag
pip install -r requirements.txt
```

## Setup

1. Copy your PDF and Word reports into the `./reports` folder.
2. Export your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

```bash
python rag_pipeline.py
```

On the first run the pipeline will index all files in `./reports`. Subsequent runs load the existing index immediately.

### Chat commands

| Input | Action |
|---|---|
| Any question | Retrieve relevant chunks and generate an answer |
| `save md` | Save the last answer as a Markdown file in `./outputs/` |
| `save docx` | Save the last answer as a Word document in `./outputs/` |
| `quit` / `exit` / `q` | Exit the program |

### Example session

```
You: Summarise how Project A and Project B validated the technology.

[Sources: ProjectA_report.pdf, ProjectB_milestone3.pdf, ...]
Claude: ## Technology Validation

### Project A
...

You: save docx
Saved → /home/user/reports-rag/outputs/summarise_how_project_a_20240315_142301.docx
```

## Configuration

Key constants at the top of `rag_pipeline.py`:

| Constant | Default | Description |
|---|---|---|
| `REPORTS_DIR` | `./reports` | Folder containing source documents |
| `CHROMA_DIR` | `./chroma_db` | Where the vector index is stored |
| `CHUNK_TOKENS` | `500` | Target chunk size (in tokens) |
| `OVERLAP_TOKENS` | `50` | Overlap between consecutive chunks |
| `TOP_K` | `15` | Number of chunks retrieved per question |
| `BOOST_K` | `3` | Extra chunks guaranteed per named project |
| `CLAUDE_MODEL` | `claude-opus-4-6` | Anthropic model used for generation |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |

## Project Keyword Boost

When a question explicitly names a project, the pipeline guarantees chunks from that project's documents are included in the context even if their semantic similarity score is low. Keywords are matched against source filenames and configured in the `PROJECT_KEYWORDS` dictionary in `rag_pipeline.py`.

To add a new project:

```python
PROJECT_KEYWORDS: dict[str, callable] = {
    "project-a": lambda s: "PROJECT_A" in s.upper(),
    "project-b": lambda s: "PROJECT_B" in s.upper(),
}
```

## Re-indexing

If you add new files to `./reports` after the first run, delete the index and rebuild:

```bash
rm -rf chroma_db
python rag_pipeline.py
```

## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Claude API client |
| `pymupdf` | PDF text extraction |
| `python-docx` | Word document text extraction |
| `chromadb` | Local vector database |
| `sentence-transformers` | Local embedding model |
| `torch` | Required by sentence-transformers |

## License

MIT
