import streamlit as st
import os
import json
import numpy as np
from typing import List, Dict, Tuple
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import requests
import re
import tiktoken
from dotenv import load_dotenv
import time

load_dotenv()

class PDFRAGSystem:
    def __init__(self, openrouter_api_key: str):
        """
        Initialize the RAG system with necessary models and configurations.
        
        Args:
            openrouter_api_key: API key for OpenRouter
        """
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Initialize embedding model (using sentence-transformers)
        with st.spinner("Loading embedding model... This may take a moment."):
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize storage
        self.chunks = []
        self.embeddings = None
        self.index = None
        
        # Tokenizer for chunk size estimation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text content from PDF file.
        
        Args:
            pdf_file: Uploaded PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
                
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
        
        return text.strip()
    
    def create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[Dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Cleaned text to chunk
            chunk_size: Target size for each chunk in tokens
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        progress_text = st.empty()
        
        for i, para in enumerate(paragraphs):
            progress_text.text(f"Processing paragraph {i+1}/{len(paragraphs)}...")
            para_tokens = len(self.tokenizer.encode(para))
            
            # If paragraph is too large, split it
            if para_tokens > chunk_size:
                sentences = para.split('. ')
                for sent in sentences:
                    sent_tokens = len(self.tokenizer.encode(sent))
                    if current_tokens + sent_tokens > chunk_size and current_chunk:
                        chunks.append({
                            'id': chunk_id,
                            'text': current_chunk.strip(),
                            'tokens': current_tokens
                        })
                        chunk_id += 1
                        
                        # Start new chunk with overlap
                        overlap_text = current_chunk.split()[-overlap:] if overlap > 0 else []
                        current_chunk = ' '.join(overlap_text) + ' ' + sent
                        current_tokens = len(self.tokenizer.encode(current_chunk))
                    else:
                        current_chunk += ' ' + sent
                        current_tokens += sent_tokens
            else:
                if current_tokens + para_tokens > chunk_size and current_chunk:
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk.strip(),
                        'tokens': current_tokens
                    })
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk.split()[-overlap:] if overlap > 0 else []
                    current_chunk = ' '.join(overlap_text) + ' ' + para
                    current_tokens = len(self.tokenizer.encode(current_chunk))
                else:
                    current_chunk += ' ' + para
                    current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'tokens': current_tokens
            })
        
        progress_text.empty()
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches for efficiency
        batch_size = 32
        all_embeddings = []
        
        progress_text = st.empty()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
            all_embeddings.append(embeddings)
            progress_text.text(f"Created embeddings for {min(i+batch_size, len(texts))}/{len(texts)} chunks...")
        
        progress_text.empty()
        return np.vstack(all_embeddings)
    
    def create_vector_index(self, embeddings: np.ndarray):
        """
        Create FAISS index for efficient similarity search.
        
        Args:
            embeddings: Numpy array of embeddings
        """
        dimension = embeddings.shape[1]
        
        # Using IndexFlatIP for inner product (similar to cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
    def process_pdf(self, pdf_file, chunk_size: int, chunk_overlap: int) -> bool:
        """
        Complete pipeline to process a PDF document.
        
        Args:
            pdf_file: Uploaded PDF file
            
        Returns:
            Success status
        """
        try:
            start_time = time.time()
            
            # Extract text
            with st.spinner("Extracting text from PDF..."):
                raw_text = self.extract_text_from_pdf(pdf_file)
            
            if not raw_text.strip():
                st.error("No text extracted from PDF. The PDF might be image-based or empty.")
                return False
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            # Create chunks
            with st.spinner("Creating chunks..."):
                self.chunks = self.create_chunks(cleaned_text, chunk_size, chunk_overlap)
            
            if not self.chunks:
                st.error("No chunks created from text. The document might be too short.")
                return False
            
            st.info(f"Created {len(self.chunks)} chunks")
            
            # Generate embeddings
            with st.spinner("Generating embeddings..."):
                self.embeddings = self.generate_embeddings(self.chunks)
            
            # Create index
            with st.spinner("Creating vector index..."):
                self.create_vector_index(self.embeddings)
            
            end_time = time.time()
            st.success(f"Successfully processed {len(self.chunks)} chunks in {end_time - start_time:.2f} seconds.")
            return True
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of (chunk, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return chunks with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):  # Safety check
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def generate_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """
        Generate answer using OpenRouter API with retrieved context.
        
        Args:
            query: User question
            relevant_chunks: Retrieved chunks with scores
            
        Returns:
            Generated answer
        """
        # Prepare context from chunks
        context = "\n\n".join([chunk['text'] for chunk, _ in relevant_chunks])
        
        # Prepare prompt
        prompt = f"""You are a helpful AI assistant. Answer the following question based ONLY on the provided context. If the context does not contain the answer, state that the context does not provide an answer. Do not use any external knowledge.

        Context:
        {context}

        Question: {query}

        Answer:"""
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "RAG PDF Q&A System"
        }
        
        data = {
            "model": "meta-llama/llama-4-scout:free",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                self.openrouter_base_url,
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('choices') and result['choices'][0].get('message') and result['choices'][0]['message'].get('content'):
                return result['choices'][0]['message']['content']
            else:
                return "API Error: Unexpected response structure from LLM."
            
        except requests.exceptions.RequestException as e:
            return f"API Request Error: {e}"
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def ask(self, query: str, top_k: int = 5) -> Dict:
        """
        Main method to ask a question and get an answer.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.index:
            return {"answer": "No PDF has been processed yet. Please process a PDF first.", "sources": []}
        
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query, top_k)
            
            if not relevant_chunks:
                return {"answer": "No relevant context found for your query in the document.", "sources": []}
            
            # Generate answer
            answer = self.generate_answer(query, relevant_chunks)
            
            # Prepare response
            sources = [
                {
                    "content": chunk['text'],
                    "score": score,
                    "index": chunk['id']
                }
                for chunk, score in relevant_chunks
            ]
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            return {"answer": f"Error during question answering: {str(e)}", "sources": []}

# Function to create a unique key for the session state
def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(int(time.time()))
    return st.session_state["session_id"]

# Initialize chat history in session state if it doesn't exist
def init_chat_history():
    session_id = get_session_id()
    if f"chat_history_{session_id}" not in st.session_state:
        st.session_state[f"chat_history_{session_id}"] = []

# Function to add a message to chat history
def add_message(role, content, sources=None):
    session_id = get_session_id()
    st.session_state[f"chat_history_{session_id}"].append({
        "role": role,
        "content": content,
        "sources": sources if sources else []
    })

# Function to get chat history
def get_chat_history():
    session_id = get_session_id()
    return st.session_state[f"chat_history_{session_id}"]

# Function to clear chat history
def clear_chat_history():
    session_id = get_session_id()
    st.session_state[f"chat_history_{session_id}"] = []

# Function to display chat messages
def display_chat():
    chat_history = get_chat_history()
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display sources if available and not empty
            if message["role"] == "assistant" and message["sources"]:
                with st.expander("View Source Chunks"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}** (Relevance Score: {source['score']:.4f})")
                        st.text(source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"])
                        st.markdown("---")

def main():
    st.set_page_config(
        page_title="PDF Chat: RAG-based Q&A",
        page_icon="📚",
        layout="wide",
    )
    
    # Initialize session states
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    
    # Initialize chat history
    init_chat_history()
    
    # Title and description
    st.title("PDF Chat: RAG-based Q&A System")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    chunk_size = 500
    chunk_overlap = 50
    top_k = 5
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    
    # PDF upload
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Process PDF button (only show if file is uploaded)
        process_button_disabled = uploaded_file is None
        if process_button_disabled:
            st.button("Process PDF", disabled=True, help="Upload a PDF first")
        else:
            if st.button("Process PDF", type="primary"):
                if not api_key:
                    st.error("OpenRouter API key not provided. Please set the OPENROUTER_API_KEY environment variable.")
                else:
                    with st.spinner("Processing PDF..."):
                        # Initialize RAG system
                        st.session_state.rag_system = PDFRAGSystem(openrouter_api_key=api_key)
                        
                        # Process the PDF
                        success = st.session_state.rag_system.process_pdf(uploaded_file, chunk_size, chunk_overlap)
                        
                        if success:
                            st.session_state.pdf_processed = True
                            # Clear chat history when a new PDF is processed
                            clear_chat_history()
                        else:
                            st.error("Failed to process PDF. Please try again or upload a different file.")
    
    with col2:
        with st.container():
            # Clear chat button
            if st.button("Clear Chat History", type="secondary", key="clear_chat_btn"):
                clear_chat_history()
                st.success("Chat history cleared!")
    
    # Add a separator below the buttons
    st.markdown("---")
    
    # Main chat interface
    display_chat()
    
    # Chat input
    if query := st.chat_input("Ask a question about the document", 
                             disabled=not st.session_state.pdf_processed):
        # Add user message to chat
        add_message("user", query)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Get bot response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Processing...")
            
            if st.session_state.rag_system:
                result = st.session_state.rag_system.ask(query, top_k=top_k)
                response_placeholder.markdown(result["answer"])
                
                # Show sources if available
                if result["sources"]:
                    with st.expander("View Source Chunks"):
                        for i, source in enumerate(result["sources"]):
                            st.markdown(f"**Source {i+1}** (Relevance Score: {source['score']:.4f})")
                            st.text(source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"])
                            st.markdown("---")
                
                # Add assistant response to chat history
                add_message("assistant", result["answer"], result["sources"])
            else:
                error_msg = "PDF not processed yet. Please upload and process a PDF first."
                response_placeholder.markdown(error_msg)
                add_message("assistant", error_msg)

if __name__ == "__main__":
    main()