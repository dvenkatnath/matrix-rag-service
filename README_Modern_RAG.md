# 🤖 Matrix Studio - Modern RAG Application

A sleek, browser-based web application that combines document ingestion and intelligent chatbot functionality in a modern, user-friendly interface. **Now powered by Pinecone for reliable cloud-based vector storage!**

## ✨ Features

### 🎨 Modern UI/UX
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Clean Interface**: Modern, minimalist design with color icons
- **Real-time Status**: Live system status indicators
- **Interactive Chat**: Modern chat interface with message bubbles
- **Drag & Drop**: Easy file upload with visual feedback

### 📁 Document Processing
- **Multi-format Support**: PDF, CSV, Excel, Word, and Text files
- **Encoding Detection**: Automatic handling of different text encodings
- **Error Handling**: Robust error handling with user-friendly messages
- **Batch Processing**: Upload and process multiple files simultaneously

### 🤖 AI-Powered Chat
- **RAG Integration**: Retrieval-Augmented Generation for accurate responses
- **Context Awareness**: Responses based on your uploaded documents
- **Streaming Responses**: Real-time response generation
- **Conversation History**: Persistent chat history within the session

### ☁️ Cloud-Based Storage
- **Pinecone Integration**: Reliable cloud-based vector database
- **Scalable**: Handles large document collections efficiently
- **Persistent**: Data persists across sessions and deployments
- **Fast Retrieval**: Optimized for similarity search performance

## 🚀 Quick Start

### 1. Set Up Pinecone (Required)

#### Create Pinecone Account
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Sign up for a free account
3. Get your API key from the console

#### Create Pinecone Index
1. In Pinecone Console, click "Create Index"
2. Set index name: `matrix-rag-index`
3. Set dimensions: `1536` (for text-embedding-3-small)
4. Set metric: `cosine`
6. Choose your preferred cloud and region
7. Click "Create Index"

### 2. Set Up Environment Variables
Create a `.env` file in your project directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run modern_rag_app.py
```

### 5. Access the Application
Open your browser and navigate to: `http://localhost:8501`

## 📖 Usage Guide

### Initial Setup
1. **Set API Keys**: Ensure both OpenAI and Pinecone API keys are configured
2. **Initialize RAG System**: Click the "🚀 Initialize RAG System" button in the sidebar
3. **Verify Status**: Check that both Vector Store and LLM show "Active" status
4. **Upload Documents**: Use the file uploader to add your documents

### Document Ingestion
1. **Select Files**: Choose one or more files from your computer
2. **Supported Formats**: PDF, CSV, Excel (.xlsx, .xls), Word (.docx, .doc), Text (.txt)
3. **Process Documents**: Click "📥 Ingest Documents" to add them to the knowledge base
4. **Monitor Progress**: Watch real-time feedback on document processing

### Chat Interface
1. **Ask Questions**: Type your questions in the chat input at the bottom
2. **Get Responses**: The AI will search your documents and provide relevant answers
3. **View History**: All conversations are saved in the current session
4. **Clear Chat**: Use the "🗑️ Clear Chat" button to start fresh

## 🏗️ Architecture

### Frontend
- **Streamlit**: Modern web framework for Python applications
- **Custom CSS**: Beautiful styling with clean, modern design
- **Responsive Layout**: Adaptive design for different screen sizes

### Backend
- **LangChain**: Framework for building LLM applications
- **Pinecone**: Cloud-based vector database for document storage and retrieval
- **OpenAI**: GPT-4o-mini for text generation and text-embedding-3-small for embeddings

### Data Flow
1. **Document Upload** → File processing and text extraction
2. **Text Chunking** → Splitting documents into manageable pieces
3. **Embedding Generation** → Converting text to vector representations
4. **Vector Storage** → Storing embeddings in Pinecone cloud database
5. **Query Processing** → Retrieving relevant document chunks
6. **Response Generation** → Generating answers using retrieved context

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `PINECONE_API_KEY`: Your Pinecone API key (required)

### Model Settings
- **Embeddings Model**: `text-embedding-3-small` (1536 dimensions)
- **LLM Model**: `gpt-4o-mini`
- **Temperature**: 0.5 (balanced creativity and accuracy)
- **Chunk Size**: 500 characters with 150 character overlap
- **Vector Database**: Pinecone cloud index

### Pinecone Configuration
- **Index Name**: `matrix-rag-index`
- **Dimensions**: 1536 (matches text-embedding-3-small)
- **Metric**: Cosine similarity
- **Text Key**: "text" (for document content)

## 📁 File Structure
```
RAG-LANG/
├── modern_rag_app.py          # Main application file
├── requirements.txt           # Dependencies
├── README_Modern_RAG.md       # This documentation
├── .env                       # Environment variables
├── data/                      # Document storage (optional)
├── ingest_deepseek.py         # Original ingestion script
└── chatbot.py                 # Original chatbot script
```

## 🎯 Key Improvements with Pinecone

### Enhanced Features
- **Cloud Storage**: No local database files to manage
- **Scalability**: Handles large document collections efficiently
- **Reliability**: 99.9% uptime with automatic backups
- **Performance**: Optimized for fast similarity search
- **Persistence**: Data survives app restarts and redeployments

### Technical Benefits
- **No Local Dependencies**: Eliminates ChromaDB compatibility issues
- **Streamlit Cloud Ready**: Perfect for cloud deployment
- **Better Error Handling**: More robust initialization and operation
- **Production Ready**: Suitable for production workloads

## 🐛 Troubleshooting

### Common Issues

**"Pinecone API key not found"**
- Ensure your `.env` file contains the correct PINECONE_API_KEY
- Check that the file is in the same directory as the application

**"Failed to initialize RAG system"**
- Verify your internet connection
- Check that both OpenAI and Pinecone API keys are valid
- Ensure your Pinecone index exists and is properly configured

**"Error loading PDF"**
- Some PDFs may have encoding issues
- The application will try alternative loaders automatically
- Check the console for specific error messages

**"No documents found"**
- Ensure you've uploaded files in supported formats
- Check that files aren't corrupted or password-protected
- Verify file extensions are correct

### Pinecone-Specific Issues

**"Index not found"**
- Create the `matrix-rag-index` in your Pinecone console
- Ensure the index has 1536 dimensions
- Check that your API key has access to the index

**"Dimension mismatch"**
- Ensure your Pinecone index has exactly 1536 dimensions
- This matches the text-embedding-3-small model output

### Performance Tips
- **Large Files**: For very large documents, consider splitting them first
- **Batch Processing**: Upload multiple smaller files rather than one large file
- **API Limits**: Be aware of OpenAI and Pinecone rate limits
- **Index Optimization**: Use appropriate Pinecone index settings for your use case

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Submitting pull requests

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with Streamlit for the web interface
- Powered by LangChain for LLM integration
- Uses Pinecone for cloud-based vector storage
- OpenAI for language models and embeddings 