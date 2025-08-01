import streamlit as st
import os
import tempfile
from pathlib import Path
import sys

# Try to import langchain components with error handling
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError as e:
    st.error(f"Failed to import langchain_openai: {e}")
    st.stop()

try:
    from langchain_chroma import Chroma
except ImportError as e:
    st.error(f"Failed to import langchain_chroma: {e}")
    st.stop()

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    st.error(f"Failed to import langchain_text_splitters: {e}")
    st.stop()

try:
    from langchain_community.document_loaders import (
        PyPDFDirectoryLoader,
        CSVLoader,
        UnstructuredExcelLoader,
        UnstructuredWordDocumentLoader,
        TextLoader,
        UnstructuredPDFLoader
    )
except ImportError as e:
    st.error(f"Failed to import document loaders: {e}")
    st.stop()

from uuid import uuid4
from dotenv import load_dotenv
import chardet
import time

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PATH = r"chroma_db"

# Custom CSS for modern UI
st.set_page_config(
    page_title="Matrix Studio - Modern RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, simplistic design
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Modern, clean design */
    .main {
        background: #f8fafc;
        min-height: 100vh;
        padding: 0;
    }
    
    .stApp {
        background: #f8fafc;
    }
    
    .chat-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        min-height: 85vh;
    }
    
    .user-message {
        background: #3b82f6;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px;
        margin: 1rem 0;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
    }
    
    .bot-message {
        background: #f8fafc;
        color: #1a202c;
        padding: 1rem 1.5rem;
        border-radius: 18px;
        margin: 1rem 0;
        max-width: 80%;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .sidebar {
        background: #ffffff;
        border-radius: 12px;
        margin: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    .status-success {
        background: #10b981;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }
    
    .status-error {
        background: #ef4444;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.2);
    }
    
    .upload-area {
        background: #f8fafc;
        border: 2px dashed #3b82f6;
        border-radius: 8px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background: #f1f5f9;
        border-color: #2563eb;
    }
    
    /* Custom button styles */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Chat input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e2e8f0;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border-radius: 8px;
        border: 2px dashed #3b82f6;
        background: #f8fafc;
    }
    
    /* Section headers */
    .section-header {
        color: #1a202c;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Main title styling */
    .main-title {
        color: #1a202c;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 0 1.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }
    
    /* Success/error messages */
    .success-message {
        background: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }
    
    .error-message {
        background: #ef4444;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

def initialize_models():
    """Initialize the LLM and embeddings models"""
    try:
        # Initialize embeddings model
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize LLM
        llm = ChatOpenAI(
            temperature=0.5, 
            model='gpt-4o-mini',
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize vector store
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings_model,
            persist_directory=CHROMA_PATH,
        )
        
        # Initialize retriever
        retriever = vector_store.as_retriever(search_kwargs={'k': 5})
        
        return llm, vector_store, retriever
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None, None

def load_documents_from_files(uploaded_files):
    """Load documents from uploaded files"""
    documents = []
    
    for uploaded_file in uploaded_files:
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension == '.pdf':
                try:
                    loader = UnstructuredPDFLoader(tmp_path)
                    docs = loader.load()
                    documents.extend(docs)
                    st.markdown(f'<div class="success-message">✅ Successfully loaded PDF: {uploaded_file.name}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error loading PDF {uploaded_file.name}: {e}</div>', unsafe_allow_html=True)
                    
            elif file_extension == '.csv':
                loader = CSVLoader(tmp_path)
                docs = loader.load()
                documents.extend(docs)
                st.markdown(f'<div class="success-message">✅ Successfully loaded CSV: {uploaded_file.name}</div>', unsafe_allow_html=True)
                
            elif file_extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(tmp_path)
                docs = loader.load()
                documents.extend(docs)
                st.markdown(f'<div class="success-message">✅ Successfully loaded Excel: {uploaded_file.name}</div>', unsafe_allow_html=True)
                
            elif file_extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(tmp_path)
                docs = loader.load()
                documents.extend(docs)
                st.markdown(f'<div class="success-message">✅ Successfully loaded Word: {uploaded_file.name}</div>', unsafe_allow_html=True)
                
            elif file_extension == '.txt':
                try:
                    # Detect encoding
                    with open(tmp_path, 'rb') as f:
                        raw_data = f.read()
                        result = chardet.detect(raw_data)
                        encoding = result['encoding']
                    
                    loader = TextLoader(tmp_path, encoding=encoding)
                    docs = loader.load()
                    documents.extend(docs)
                    st.markdown(f'<div class="success-message">✅ Successfully loaded Text: {uploaded_file.name}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error loading text file {uploaded_file.name}: {e}</div>', unsafe_allow_html=True)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.markdown(f'<div class="error-message">❌ Error processing {uploaded_file.name}: {e}</div>', unsafe_allow_html=True)
    
    return documents

def ingest_documents(documents):
    """Ingest documents into the vector store"""
    if not documents:
        st.warning("No documents to ingest.")
        return False
    
    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_documents(documents)
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        
        # Add to vector store
        st.session_state.vector_store.add_documents(
            documents=chunks,
            ids=uuids,
        )
        
        return True
    except Exception as e:
        st.markdown(f'<div class="error-message">Error ingesting documents: {e}</div>', unsafe_allow_html=True)
        return False

def get_response(user_message):
    """Get response from the RAG system"""
    try:
        # Retrieve relevant documents
        docs = st.session_state.retriever.invoke(user_message)
        
        # Concatenate retrieved knowledge
        knowledge = "\n\n".join(doc.page_content for doc in docs)
        
        # Create prompt
        rag_prompt = f"""
        You are an assistant answering questions based on the provided knowledge.
        You must answer only using the "The knowledge" section, without adding any external information.
        You should not mention that the knowledge was retrieved.
        
        The question: {user_message}
        
        The knowledge: {knowledge}
        """
        
        # Get response from LLM
        response = st.session_state.llm.invoke(rag_prompt)
        return response.content
        
    except Exception as e:
        return f"Error generating response: {e}"

# Main application
def main():
    # No main header - removed for more space
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="main-title">🤖 Matrix - RAG Service</div>', unsafe_allow_html=True)
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            st.markdown('<div class="error-message">❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.</div>', unsafe_allow_html=True)
            return
        
        # Initialize models
        if st.button("🚀 Initialize RAG System", type="primary"):
            with st.spinner("Initializing models..."):
                llm, vector_store, retriever = initialize_models()
                if llm and vector_store and retriever:
                    st.session_state.llm = llm
                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = retriever
                    st.markdown('<div class="success-message">✅ RAG system initialized successfully!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-message">❌ Failed to initialize RAG system</div>', unsafe_allow_html=True)
        
        # System status
        st.markdown('<div class="section-header">📊 System Status</div>', unsafe_allow_html=True)
        if st.session_state.vector_store:
            st.markdown('<div class="status-success">✅ Vector Store: Active</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ Vector Store: Not Initialized</div>', unsafe_allow_html=True)
        
        if st.session_state.llm:
            st.markdown('<div class="status-success">✅ LLM: Active</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ LLM: Not Initialized</div>', unsafe_allow_html=True)
        
        # Document ingestion section
        st.markdown('<div class="section-header">📁 Document Ingestion</div>', unsafe_allow_html=True)
        st.markdown("Upload documents to add them to the knowledge base:")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'csv', 'xlsx', 'xls', 'docx', 'doc', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, CSV, Excel, Word, Text"
        )
        
        if uploaded_files and st.button("📥 Ingest Documents", type="secondary"):
            if not st.session_state.vector_store:
                st.markdown('<div class="error-message">Please initialize the RAG system first.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Processing documents..."):
                    documents = load_documents_from_files(uploaded_files)
                    if documents:
                        if ingest_documents(documents):
                            st.markdown(f'<div class="success-message">✅ Successfully ingested {len(documents)} documents!</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-message">❌ Failed to ingest documents</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No valid documents found.")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # Main content area - Full width chat
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.retriever:
            st.markdown('<div class="error-message">Please initialize the RAG system first.</div>', unsafe_allow_html=True)
        else:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
            
            # Get and display bot response
            with st.spinner("Thinking..."):
                response = get_response(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(f'<div class="bot-message">{response}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 