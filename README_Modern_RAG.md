# 🤖 Matrix Studio - Modern RAG Application

A sleek, browser-based web application that combines document ingestion and intelligent chatbot functionality in a modern, user-friendly interface.

## ✨ Features

### 🎨 Modern UI/UX
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Gradient Headers**: Beautiful visual design with modern color schemes
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

### 📊 Analytics & Monitoring
- **Document Count**: Real-time statistics on ingested documents
- **System Status**: Live monitoring of vector store and LLM status
- **Activity Tracking**: Recent query history and usage statistics

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_modern.txt
```

### 2. Set Up Environment
Create a `.env` file in your project directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Application
```bash
streamlit run modern_rag_app.py
```

### 4. Access the Application
Open your browser and navigate to: `http://localhost:8501`

## 📖 Usage Guide

### Initial Setup
1. **Initialize RAG System**: Click the "🚀 Initialize RAG System" button in the sidebar
2. **Verify Status**: Check that both Vector Store and LLM show "Active" status
3. **Upload Documents**: Use the file uploader to add your documents

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
- **Custom CSS**: Beautiful styling with gradients and modern design
- **Responsive Layout**: Adaptive design for different screen sizes

### Backend
- **LangChain**: Framework for building LLM applications
- **ChromaDB**: Vector database for document storage and retrieval
- **OpenAI**: GPT-4o-mini for text generation and text-embedding-3-small for embeddings

### Data Flow
1. **Document Upload** → File processing and text extraction
2. **Text Chunking** → Splitting documents into manageable pieces
3. **Embedding Generation** → Converting text to vector representations
4. **Vector Storage** → Storing embeddings in ChromaDB
5. **Query Processing** → Retrieving relevant document chunks
6. **Response Generation** → Generating answers using retrieved context

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Model Settings
- **Embeddings Model**: `text-embedding-3-small` (1536 dimensions)
- **LLM Model**: `gpt-4o-mini`
- **Temperature**: 0.5 (balanced creativity and accuracy)
- **Chunk Size**: 500 characters with 150 character overlap

### Customization
You can modify the following in `modern_rag_app.py`:
- **Chunk Size**: Change `chunk_size` and `chunk_overlap` in the text splitter
- **Retrieval Count**: Modify `search_kwargs={'k': 5}` for more/fewer retrieved documents
- **Temperature**: Adjust the LLM temperature for different response styles
- **UI Colors**: Modify the CSS variables for custom styling

## 📁 File Structure
```
RAG-LANG/
├── modern_rag_app.py          # Main application file
├── requirements_modern.txt    # Dependencies for the modern app
├── README_Modern_RAG.md       # This documentation
├── .env                       # Environment variables
├── chroma_db/                 # Vector database storage
├── data/                      # Document storage (optional)
├── ingest_deepseek.py         # Original ingestion script
└── chatbot.py                 # Original chatbot script
```

## 🎯 Key Differences from Original

### Enhanced Features
- **Unified Interface**: Document ingestion and chat in one application
- **Modern UI**: Beautiful, responsive design with custom styling
- **Real-time Feedback**: Live status updates and progress indicators
- **Better Error Handling**: User-friendly error messages and recovery
- **Session Management**: Persistent chat history and system state

### Technical Improvements
- **Streamlit Framework**: More modern than Gradio for web applications
- **Better File Handling**: Improved document processing with encoding detection
- **State Management**: Proper session state handling for better UX
- **Modular Design**: Cleaner code structure with separate functions

## 🐛 Troubleshooting

### Common Issues

**"OpenAI API key not found"**
- Ensure your `.env` file contains the correct API key
- Check that the file is in the same directory as the application

**"Failed to initialize RAG system"**
- Verify your internet connection
- Check that your OpenAI API key is valid and has sufficient credits
- Ensure all dependencies are installed correctly

**"Error loading PDF"**
- Some PDFs may have encoding issues
- The application will try alternative loaders automatically
- Check the console for specific error messages

**"No documents found"**
- Ensure you've uploaded files in supported formats
- Check that files aren't corrupted or password-protected
- Verify file extensions are correct

### Performance Tips
- **Large Files**: For very large documents, consider splitting them first
- **Batch Processing**: Upload multiple smaller files rather than one large file
- **Memory Usage**: Monitor system resources when processing many documents
- **API Limits**: Be aware of OpenAI rate limits for large document sets

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
- Uses ChromaDB for vector storage
- OpenAI for language models and embeddings 