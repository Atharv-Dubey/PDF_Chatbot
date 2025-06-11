# PDF-Chatbot

A lightweight RAG-based chatbot that answers questions from user-uploaded PDFs using `google/flan-t5-base`. Built for Google Colab with memory optimizations.

## Overview

PDF-Chatbot extracts text from a PDF, builds a knowledge base using Retrieval-Augmented Generation (RAG), and answers user questions about the content. It’s optimized for Colab’s resource limits.

## Features

- Upload and process PDFs.
- Ask questions about PDF content.
- Answers generated with `google/flan-t5-base`.
- Low memory usage on Colab.

## Technologies

- Python 3.7+
- PyPDF2 (PDF extraction)
- langchain (RAG)
- sentence-transformers (embeddings)
- FAISS (vector store)
- transformers (`google/flan-t5-base`)
- Google Colab

## Usage

1. Upload a PDF when prompted.
2. Wait for the chatbot to process the PDF.
3. Ask questions (e.g., "What is the main topic?").
4. Type `exit` to stop.

## How It Works

1. **Text Extraction**: Extracts PDF text using `PyPDF2`.
2. **Chunking**: Splits text into 500-character chunks.
3. **Embedding**: Converts chunks to embeddings with `sentence-transformers/all-MiniLM-L6-v2`, stored in FAISS.
4. **RAG**: Retrieves relevant chunks for a query and generates answers using `google/flan-t5-base`.

## Optimizations

- Small chunks and batch processing for FAISS.
- Memory management with garbage collection.
- Lightweight model (`google/flan-t5-base`, \~1-2GB RAM).

## Limitations

- Limited RAM on Colab (\~12GB) may struggle with large PDFs.
- `google/flan-t5-base` lacks `temperature` for answer randomness.
- PDFs with images or complex formatting may not extract well.

## 