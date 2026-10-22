@'
# DocuMind AI

A local-first Retrieval-Augmented Generation (RAG) application for asking questions about technical documentation.

DocuMind AI combines semantic search with a locally running language model to generate answers grounded in retrieved document passages.

## Features

- Document ingestion for TXT, Markdown, HTML, and AsciiDoc files
- Text chunking with configurable overlap
- Semantic embeddings using Sentence Transformers
- Persistent vector storage with ChromaDB
- Local answer generation through Ollama
- Source filenames displayed alongside answers
- Retrieval evaluation using Hit Rate@K

## Technology Stack

- Python
- Sentence Transformers: `all-MiniLM-L6-v2`
- ChromaDB
- Ollama
- Qwen3 local model
- Beautiful Soup
- Requests

## Architecture

1. Load technical documents from the local `documents/` directory.
2. Extract and normalize their text.
3. Split the text into overlapping chunks.
4. Generate embeddings and store them in ChromaDB.
5. Embed a user's question and retrieve the closest chunks.
6. Send the retrieved context to the local Ollama model.
7. Display the generated answer and retrieved source filenames.

## Requirements

- Python compatible with the installed dependencies
- Ollama installed and running locally
- The local Ollama model configured as `qwen3-local`
- A local collection of supported documentation files

## Setup

Clone the repository and enter the project directory:

@'
# DocuMind AI

A local-first Retrieval-Augmented Generation (RAG) application for asking questions about technical documentation.

DocuMind AI combines semantic search with a locally running language model to generate answers grounded in retrieved document passages.

## Features

- Document ingestion for TXT, Markdown, HTML, and AsciiDoc files
- Text chunking with configurable overlap
- Semantic embeddings using Sentence Transformers
- Persistent vector storage with ChromaDB
- Local answer generation through Ollama
- Source filenames displayed alongside answers
- Retrieval evaluation using Hit Rate@K

## Technology Stack

- Python
- Sentence Transformers: `all-MiniLM-L6-v2`
- ChromaDB
- Ollama
- Qwen3 local model
- Beautiful Soup
- Requests

## Architecture

1. Load technical documents from the local `documents/` directory.
2. Extract and normalize their text.
3. Split the text into overlapping chunks.
4. Generate embeddings and store them in ChromaDB.
5. Embed a user's question and retrieve the closest chunks.
6. Send the retrieved context to the local Ollama model.
7. Display the generated answer and retrieved source filenames.

## Requirements

- Python compatible with the installed dependencies
- Ollama installed and running locally
- The local Ollama model configured as `qwen3-local`
- A local collection of supported documentation files

## Setup

Clone the repository and enter the project directory:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd documind-ai
