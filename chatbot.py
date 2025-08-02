from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Initialize the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize the LLM model with lower temperature for more consistent answers
llm = ChatOpenAI(temperature=0.1, model='gpt-4o-mini')

# Connect to the ChromaDB vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the retriever with better parameters for more accurate answers
retriever = vector_store.as_retriever(
    search_kwargs={
        'k': 15,  # Retrieve more documents for better coverage and summarization
        'fetch_k': 30,  # Fetch more candidates before selecting top k
        'lambda_mult': 0.3  # Balance between similarity and diversity
    }
)

# Function for streaming responses
def stream_response(message, history):
    # Retrieve relevant chunks based on the question with improved strategy
    docs = retriever.invoke(message)
    
    # Filter and rank documents by relevance
    if docs:
        # Sort by metadata score if available, otherwise keep original order
        try:
            docs = sorted(docs, key=lambda x: getattr(x, 'metadata', {}).get('score', 0), reverse=True)
        except:
            pass
    
    # Concatenate retrieved knowledge with source information
    knowledge_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content
        knowledge_parts.append(f"[Source: {source}]\n{content}")
    
    knowledge = "\n\n".join(knowledge_parts)

    # Ensure there's a user message
    if message:
        partial_message = ""

        rag_prompt = f"""
        You are an expert assistant that can answer questions, provide summaries, and analyze information based on the provided knowledge.
        
        RETRIEVAL ANALYSIS:
        - Number of sources retrieved: {len(docs)}
        - Sources: {', '.join(set([doc.metadata.get('source', 'Unknown') for doc in docs]))}
        
        IMPORTANT INSTRUCTIONS:
        1. Answer ONLY using the information provided in "The knowledge" section
        2. If the knowledge doesn't contain enough information, say "I don't have enough information to answer this question based on the available documents."
        3. Be specific and accurate in your answers
        4. Do not add any external information or assumptions
        5. Do not mention that the knowledge was retrieved from documents
        6. If asked about a person, provide their full name, role, and relevant details
        7. If asked to summarize, create a comprehensive summary using all relevant information from the knowledge
        8. If asked for a specific number of sentences, try to match that requirement
        9. Search carefully through all the provided knowledge for relevant information
        10. Provide detailed and comprehensive responses when information is available
        11. Consider the source of information when providing answers
        12. If multiple sources contain conflicting information, mention this and provide the most relevant details
        
        The question: {message}
        
        Conversation history: {history}
        
        The knowledge: {knowledge}
        
        Provide a comprehensive answer based on the knowledge provided above:
        """

        # Stream response to the Gradio UI
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# Initialize the Gradio chatbot (No "Chatbot" text, Custom title)
chatbot = gr.ChatInterface(
    fn=stream_response,
    title="Matrix Studio - RAG service",  # ✅ Correct way to set title
    #theme="soft",  # Optional: Smooth theme for better UI
    type="messages",  # ✅ Fix for the Gradio warning
    textbox=gr.Textbox(
        placeholder="Ask a question...",
        container=False,
        autoscroll=True,
        scale=7
    ),
)

# Launch the chatbot
chatbot.launch(share=True)
