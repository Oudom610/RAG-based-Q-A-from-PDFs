# PDF Chat: RAG-based Q&A System

Interactive Streamlit application that lets you upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG).

## What It Does

1. **Extracts text** from uploaded PDF documents
2. **Creates embeddings** using sentence-transformers for semantic search
3. **Stores vectors** in FAISS index for fast similarity search
4. **Answers questions** using retrieved context and OpenRouter LLM API
5. **Chat interface** with conversation history and source citations

## Features

- **Smart chunking**: Splits documents with configurable overlap for better context
- **Semantic search**: Uses `all-MiniLM-L6-v2` embeddings for relevance matching
- **Source citations**: Shows which document chunks were used for each answer
- **Chat history**: Maintains conversation context within sessions
- **Real-time processing**: Live progress indicators during PDF processing
- **Error handling**: Graceful fallbacks for API issues and malformed PDFs

## Installation

```bash
pip install streamlit sentence-transformers faiss-cpu PyPDF2 requests tiktoken python-dotenv numpy
```

## Setup

### 1. Get OpenRouter API Key
Sign up at [OpenRouter](https://openrouter.ai/) and get your API key

### 2. Set Environment Variable
Create a `.env` file:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Run the Application
```bash
streamlit run official_rag_app.py
```

## How to Use

1. **Upload PDF**: Click "Upload a PDF document" and select your file
2. **Process Document**: Click "Process PDF" to extract and index content
3. **Ask Questions**: Use the chat input to ask questions about the document
4. **View Sources**: Expand "View Source Chunks" to see supporting text

## Configuration

Default settings in the code:
```python
chunk_size = 500        # Tokens per chunk
chunk_overlap = 50      # Overlap between chunks
top_k = 5              # Number of relevant chunks to retrieve
model = "meta-llama/llama-4-scout:free"  # OpenRouter model
```

## System Architecture

### Processing Pipeline:
1. **PDF Text Extraction** → PyPDF2
2. **Text Cleaning** → Remove artifacts, fix hyphenation
3. **Chunking** → Split by paragraphs with token-based sizing
4. **Embedding Generation** → sentence-transformers model
5. **Vector Indexing** → FAISS for similarity search

### Query Pipeline:
1. **Query Embedding** → Convert question to vector
2. **Similarity Search** → Find most relevant chunks
3. **Context Assembly** → Combine retrieved text
4. **Answer Generation** → OpenRouter LLM with context
5. **Response Display** → Show answer with sources

## Key Components

- **`PDFRAGSystem`** - Main RAG implementation class
- **`extract_text_from_pdf()`** - PDF text extraction
- **`create_chunks()`** - Text segmentation with overlap
- **`generate_embeddings()`** - Vector creation
- **`retrieve_relevant_chunks()`** - Similarity search
- **`generate_answer()`** - LLM response generation

## Features in Detail

### Smart Text Processing
- Removes PDF extraction artifacts
- Handles hyphenated words across lines
- Preserves paragraph structure
- Token-aware chunking

### Efficient Search
- FAISS vector index for fast retrieval
- Cosine similarity scoring
- Configurable result count
- Source chunk tracking

### Interactive Interface
- Real-time processing feedback
- Conversation history
- Source citation expansion
- Clear chat functionality

## Requirements

- Python 3.7+
- Streamlit
- sentence-transformers
- faiss-cpu
- PyPDF2
- OpenRouter API access

## Performance

- **Processing**: ~30 seconds for typical 50-page PDF
- **Search**: Sub-second response times
- **Memory**: Efficient batch processing for large documents
- **Scalability**: FAISS enables handling thousands of chunks

## Limitations

- Text-based PDFs only (no OCR for images)
- OpenRouter API dependency for answer generation
- Memory usage scales with document size
- English language optimized

## License

Open source - modify as needed for your use case.
