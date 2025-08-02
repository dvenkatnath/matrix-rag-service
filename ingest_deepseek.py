from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = r"data"  # Ensure this path contains PDFs, CSVs, Excel, etc.
CHROMA_PATH = r"chroma_db"

# Note: Using OpenAI embeddings since Deepseek doesn't have a dedicated embeddings endpoint

# Ensure OpenAI API Key is set
if "OPENAI_API_KEY" not in os.environ or not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API key is missing. Check your .env file.")

# Initiate the embeddings model with OpenAI
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Test API with a sample embedding request
try:
    test_embeddings = embeddings_model.embed_documents(["Test sentence"])
    if not test_embeddings or len(test_embeddings) == 0:
        raise ValueError("Deepseek embeddings returned an empty result. Check API limits.")
except Exception as e:
    raise ValueError(f"Error with OpenAI embeddings: {e}")

# Initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Function to load documents from all supported formats
def load_documents(directory: str):
    """Loads documents from various formats in the given directory."""
    documents = []

    # Load PDFs with error handling
    try:
        pdf_loader = PyPDFDirectoryLoader(directory)
        pdf_documents = pdf_loader.load()
        documents.extend(pdf_documents)
        print(f"Successfully loaded {len(pdf_documents)} PDF documents")
    except Exception as e:
        print(f"Warning: Error loading PDFs from {directory}: {e}")
        # Try loading PDFs individually with different loaders
        for pdf_file in Path(directory).rglob("*.pdf"):
            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(str(pdf_file))
                pdf_docs = loader.load()
                documents.extend(pdf_docs)
                print(f"Successfully loaded PDF: {pdf_file}")
            except Exception as e2:
                print(f"Error loading PDF {pdf_file}: {e2}")
                continue

    # Load CSVs with error handling
    for file in Path(directory).rglob("*.csv"):
        try:
            loader = CSVLoader(str(file))
            csv_docs = loader.load()
            documents.extend(csv_docs)
            print(f"Successfully loaded CSV: {file}")
        except Exception as e:
            print(f"Error loading CSV {file}: {e}")

    # Load Excel files with error handling
    for file in Path(directory).rglob("*.xlsx"):
        try:
            loader = UnstructuredExcelLoader(str(file))
            excel_docs = loader.load()
            documents.extend(excel_docs)
            print(f"Successfully loaded Excel: {file}")
        except Exception as e:
            print(f"Error loading Excel {file}: {e}")

    # Load Word files with error handling
    for file in Path(directory).rglob("*.docx"):
        try:
            loader = UnstructuredWordDocumentLoader(str(file))
            word_docs = loader.load()
            documents.extend(word_docs)
            print(f"Successfully loaded Word: {file}")
        except Exception as e:
            print(f"Error loading Word {file}: {e}")

    # Load Text files with encoding detection and error handling
    for file in Path(directory).rglob("*.txt"):
        try:
            # Try to detect encoding
            import chardet
            with open(file, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            
            # Load with detected encoding
            loader = TextLoader(str(file), encoding=encoding)
            text_docs = loader.load()
            documents.extend(text_docs)
            print(f"Successfully loaded Text: {file} (encoding: {encoding})")
        except Exception as e:
            print(f"Warning: Could not load {file} with detected encoding. Trying utf-8: {e}")
            try:
                # Fallback to utf-8
                loader = TextLoader(str(file), encoding='utf-8')
                text_docs = loader.load()
                documents.extend(text_docs)
                print(f"Successfully loaded Text: {file} (encoding: utf-8)")
            except Exception as e2:
                print(f"Error: Could not load {file} with utf-8 encoding either: {e2}")
                continue

    print(f"\n📊 Document Loading Summary:")
    print(f"Total documents loaded: {len(documents)}")
    
    # Count by file type
    file_types = {}
    for doc in documents:
        source = doc.metadata.get('source', 'Unknown')
        ext = Path(source).suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    
    for ext, count in file_types.items():
        print(f"  {ext}: {count} documents")
    
    return documents

# Load and process documents
raw_documents = load_documents(DATA_PATH)

# Split the text into smaller chunks with better parameters for accuracy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Smaller chunks for more precise retrieval
    chunk_overlap=100,  # Good overlap to maintain context
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Better separators
)

chunks = text_splitter.split_documents(raw_documents)
uuids = [str(uuid4()) for _ in range(len(chunks))]

# Add to ChromaDB
try:
    vector_store.add_documents(
        documents=chunks,
        ids=uuids,
    )
    print("Successfully added documents to the vector store.")
except Exception as e:
    raise ValueError(f"Error adding documents to vector store: {e}")
