import streamlit as st
import os
import tempfile
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredPDFLoader
)
from uuid import uuid4
from dotenv import load_dotenv
import chardet
import time

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PATH = "chroma_db"

# Page config
st.set_page_config(
    page_title="Matrix RAG Service",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        min-height: 400px;
    }
    
    .user-message {
        background: #f8f9fa;
        color: #495057;
        padding: 0.75rem 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
        border: 1px solid #e9ecef;
    }
    
    .bot-message {
        background: white;
        color: #333;
        padding: 0.75rem 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        border: 1px solid #e9ecef;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
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
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
            return None, None, None
        
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
            collection_name="matrix_collection",
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
                    st.success(f"✅ Successfully loaded PDF: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error loading PDF {uploaded_file.name}: {e}")
                    
            elif file_extension == '.csv':
                loader = CSVLoader(tmp_path)
                docs = loader.load()
                documents.extend(docs)
                st.success(f"✅ Successfully loaded CSV: {uploaded_file.name}")
                
            elif file_extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(tmp_path)
                docs = loader.load()
                documents.extend(docs)
                st.success(f"✅ Successfully loaded Excel: {uploaded_file.name}")
                
            elif file_extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(tmp_path)
                docs = loader.load()
                documents.extend(docs)
                st.success(f"✅ Successfully loaded Word: {uploaded_file.name}")
                
            elif file_extension == '.txt':
                try:
                    # Try to detect encoding
                    with open(tmp_path, 'rb') as f:
                        raw_data = f.read()
                        result = chardet.detect(raw_data)
                        encoding = result['encoding']
                    
                    loader = TextLoader(tmp_path, encoding=encoding)
                    docs = loader.load()
                    documents.extend(docs)
                    st.success(f"✅ Successfully loaded Text: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error loading text file {uploaded_file.name}: {e}")
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
    
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
        
        st.success(f"✅ Successfully ingested {len(chunks)} document chunks")
        return True
    except Exception as e:
        st.error(f"Error ingesting documents: {e}")
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
    # Sidebar
    with st.sidebar:
        st.markdown("### 🤖 Matrix - RAG Service")
        st.markdown("### ⚙️ Configuration")
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            st.markdown('<div class="status-error">❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.</div>', unsafe_allow_html=True)
            return
        
        # Initialize models
        if st.button("🚀 Initialize RAG System", type="primary"):
            with st.spinner("Initializing models..."):
                llm, vector_store, retriever = initialize_models()
                if llm and vector_store and retriever:
                    st.session_state.llm = llm
                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = retriever
                    st.markdown('<div class="status-success">✅ RAG system initialized successfully!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-error">❌ Failed to initialize RAG system</div>', unsafe_allow_html=True)
        
        # System status
        st.markdown("### 📊 System Status")
        if st.session_state.vector_store:
            st.markdown('<div class="status-success">✅ Vector Store: Active</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ Vector Store: Not Initialized</div>', unsafe_allow_html=True)
            
        if st.session_state.llm:
            st.markdown('<div class="status-success">✅ LLM: Active</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ LLM: Not Initialized</div>', unsafe_allow_html=True)
        
        # Document ingestion
        st.markdown("### 📁 Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'csv', 'xlsx', 'xls', 'docx', 'doc', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, CSV, Excel, Word, or Text files"
        )
        
        if uploaded_files and st.button("📥 Ingest Documents", type="secondary"):
            if not st.session_state.vector_store:
                st.markdown('<div class="status-error">Please initialize the RAG system first.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Processing documents..."):
                    documents = load_documents_from_files(uploaded_files)
                    if documents:
                        success = ingest_documents(documents)
                        if success:
                            st.markdown('<div class="status-success">✅ Documents ingested successfully!</div>', unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main content area - Full width chat
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                # Bot message
                st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
                
                # Copy button after the message
                message_content = message["content"].replace('"', '\\"').replace('\n', '\\n')
                copy_script = f"""
                <script>
                function copyText_{i}() {{
                    navigator.clipboard.writeText("{message_content}");
                    // Show feedback
                    const button = document.getElementById('copy-btn-{i}');
                    button.innerHTML = '✅';
                    setTimeout(() => {{
                        button.innerHTML = '⏷';
                    }}, 1000);
                }}
                </script>
                """
                st.markdown(copy_script, unsafe_allow_html=True)
                
                # Copy button positioned after the message
                st.markdown(f'<div style="text-align: right; margin-top: 5px;"><button id="copy-btn-{i}" onclick="copyText_{i}()" style="background: none; border: none; cursor: pointer; font-size: 16px; padding: 5px; color: #6c757d;" title="Copy response">⏷</button></div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.llm or not st.session_state.retriever:
            st.error("Please initialize the RAG system first.")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get response
            with st.spinner("Generating response..."):
                response = get_response(prompt)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to display new messages
            st.rerun()

if __name__ == "__main__":
    main() 