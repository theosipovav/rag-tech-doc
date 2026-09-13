"""
Streamlit Frontend for RAG Chat System.
Provides UI for document upload and chat interface.
"""
import os
import time
import requests
from typing import List, Optional
from datetime import datetime

import streamlit as st

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
UPLOAD_DIR = "./data/uploads"

# Page config
st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .source-card {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .stats-box {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_system_stats() -> dict:
    """Get system statistics from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


def upload_file(file) -> dict:
    """Upload a file to the API."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=300)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def chat_with_rag(message: str, top_k: int = 5) -> dict:
    """Send a message to the chat API."""
    try:
        payload = {
            "message": message,
            "top_k": top_k,
            "include_sources": True,
            "conversation_history": []
        }
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=120)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def semantic_search(query: str, top_k: int = 10) -> dict:
    """Perform semantic search."""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "min_score": 0.0
        }
        response = requests.post(f"{API_BASE_URL}/search", json=payload, timeout=60)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def reset_database():
    """Reset the vector database."""
    try:
        response = requests.delete(f"{API_BASE_URL}/reset", timeout=30)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.title("RAG PDF Chat")
        st.markdown("---")
        
        # API Status
        api_healthy = check_api_health()
        if api_healthy:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.warning(f"Expected at: {API_BASE_URL}")
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["💬 Chat", "📤 Upload Documents", "🔍 Search", "⚙️ Settings"],
            index=0
        )
        
        st.markdown("---")
        
        # Quick stats
        if api_healthy:
            stats = get_system_stats()
            if stats:
                st.markdown("### System Status")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Vector DB",
                        "✅" if stats.get("vector_db_connected") else "❌"
                    )
                with col2:
                    st.metric(
                        "Embeddings",
                        "✅" if stats.get("embedding_model") else "❌"
                    )
                
                col3, col4 = st.columns(2)
                with col3:
                    st.metric(
                        "Reranker",
                        "✅" if stats.get("reranker_loaded") else "❌"
                    )
                with col4:
                    st.metric(
                        "LLM",
                        "✅" if stats.get("llm_available") else "❌"
                    )
    
    # Main content based on navigation
    if page == "💬 Chat":
        render_chat_page()
    elif page == "📤 Upload Documents":
        render_upload_page()
    elif page == "🔍 Search":
        render_search_page()
    elif page == "⚙️ Settings":
        render_settings_page()


def render_chat_page():
    """Render the chat interface."""
    st.markdown('<p class="main-header">💬 Chat with Your Documents</p>', unsafe_allow_html=True)
    st.markdown("Ask questions about your indexed PDF documents.")
    
    # Check API health
    if not check_api_health():
        st.error("API is not available. Please ensure the backend server is running.")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}**")
                        st.markdown(f"- File: `{source.get('metadata', {}).get('source_file', 'Unknown')}`")
                        if source.get('metadata', {}).get('page_number'):
                            st.markdown(f"- Page: {source['metadata']['page_number']}")
                        st.markdown(f"- Score: {source.get('score', 0):.3f}")
                        st.markdown(f"> {source.get('text', '')[:300]}...")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                response = chat_with_rag(prompt, top_k=5)
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                    answer = "Sorry, I encountered an error."
                    sources = []
                else:
                    answer = response.get("answer", "No answer generated.")
                    sources = response.get("sources", [])
                    processing_time = response.get("processing_time_ms", 0)
                    
                    st.markdown(answer)
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}**")
                                st.markdown(f"- File: `{source.get('metadata', {}).get('source_file', 'Unknown')}`")
                                if source.get('metadata', {}).get('page_number'):
                                    st.markdown(f"- Page: {source['metadata']['page_number']}")
                                st.markdown(f"- Score: {source.get('score', 0):.3f}")
                                st.markdown(f"> {source.get('text', '')[:300]}...")
                    
                    # Show processing time
                    st.caption(f"Response generated in {processing_time:.0f}ms")
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        
        # Rerun to update display
        st.rerun()


def render_upload_page():
    """Render the document upload page."""
    st.markdown('<p class="main-header">📤 Upload & Index Documents</p>', unsafe_allow_html=True)
    st.markdown("Upload PDF documents to be indexed and made searchable.")
    
    if not check_api_health():
        st.error("API is not available. Please ensure the backend server is running.")
        return
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to upload and index"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} file(s)")
        
        if st.button("📊 Upload and Index", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Uploading {file.name}...")
                
                # Upload file
                result = upload_file(file)
                results.append({
                    "filename": file.name,
                    "result": result
                })
                
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
            
            status_text.text("Upload complete!")
            
            # Display results
            st.success("Upload completed!")
            
            for result in results:
                if "error" in result["result"]:
                    st.error(f"❌ {result['filename']}: {result['result']['error']}")
                else:
                    st.success(
                        f"✅ {result['filename']}: "
                        f"Indexed {result['result'].get('chunks_indexed', 0)} chunks"
                    )


def render_search_page():
    """Render the semantic search page."""
    st.markdown('<p class="main-header">🔍 Semantic Search</p>', unsafe_allow_html=True)
    st.markdown("Search through indexed documents without generating an answer.")
    
    if not check_api_health():
        st.error("API is not available. Please ensure the backend server is running.")
        return
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Search Query", placeholder="Enter your search terms...")
    with col2:
        top_k = st.number_input("Results", min_value=1, max_value=20, value=10)
    
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching..."):
            results = semantic_search(query, top_k=top_k)
            
            if "error" in results:
                st.error(f"Error: {results['error']}")
            elif "results" in results:
                st.success(f"Found {results['total_found']} relevant chunks")
                
                for i, result in enumerate(results["results"], 1):
                    with st.container():
                        st.markdown(f"**Result {i}** (Score: {result.get('score', 0):.3f})")
                        st.markdown(f"- File: `{result.get('metadata', {}).get('source_file', 'Unknown')}`")
                        if result.get('metadata', {}).get('page_number'):
                            st.markdown(f"- Page: {result['metadata']['page_number']}")
                        st.markdown(f"> {result.get('text', '')}")
                        st.markdown("---")


def render_settings_page():
    """Render the settings page."""
    st.markdown('<p class="main-header">⚙️ Settings</p>', unsafe_allow_html=True)
    st.markdown("System configuration and maintenance.")
    
    if not check_api_health():
        st.error("API is not available. Please ensure the backend server is running.")
        return
    
    # System stats
    st.markdown("### System Statistics")
    stats = get_system_stats()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Initialized",
                "✅" if stats.get("initialized") else "❌"
            )
        with col2:
            st.metric(
                "Vector DB",
                "✅" if stats.get("vector_db_connected") else "❌"
            )
        with col3:
            st.metric(
                "Embedding Model",
                "✅" if stats.get("embedding_model") else "❌"
            )
        with col4:
            st.metric(
                "LLM Available",
                "✅" if stats.get("llm_available") else "❌"
            )
        
        # Vector DB stats
        if "vector_db_stats" in stats:
            db_stats = stats["vector_db_stats"]
            st.markdown("### Vector Database Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Collection", db_stats.get("collection_name", "N/A"))
            with col2:
                st.metric("Vectors", db_stats.get("vectors_count", 0))
    
    st.markdown("---")
    
    # Danger zone
    st.markdown("### ⚠️ Danger Zone")
    st.warning("These actions cannot be undone!")
    
    if st.button("🗑️ Reset Database", type="secondary"):
        if st.confirm("Are you sure? This will delete all indexed documents."):
            with st.spinner("Resetting database..."):
                result = reset_database()
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success("Database reset successfully!")
                    # Clear chat history
                    st.session_state.messages = []
                    st.rerun()
    
    st.markdown("---")
    
    # API Configuration
    st.markdown("### API Configuration")
    st.code(f"API Base URL: {API_BASE_URL}", language="bash")
    
    st.info(
        "To change the API URL, set the environment variable:\n"
        "```bash\nexport API_BASE_URL=http://your-api-url:8000\n```"
    )


if __name__ == "__main__":
    main()
