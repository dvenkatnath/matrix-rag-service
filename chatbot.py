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
        'k': 8,  # Retrieve more documents for better coverage
        'score_threshold': 0.7  # Only retrieve highly relevant documents
    }
)

# Function for streaming responses
def stream_response(message, history):
    # Retrieve relevant chunks based on the question
    docs = retriever.invoke(message)

    # Concatenate retrieved knowledge
    knowledge = "\n\n".join(doc.page_content for doc in docs)

    # Ensure there's a user message
    if message:
        partial_message = ""

        rag_prompt = f"""
        You are an expert assistant answering questions based on the provided knowledge.
        
        IMPORTANT INSTRUCTIONS:
        1. Answer ONLY using the information provided in "The knowledge" section
        2. If the knowledge doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the available documents."
        3. Be specific and accurate in your answers
        4. Do not add any external information or assumptions
        5. Do not mention that the knowledge was retrieved from documents
        6. If asked about a person, provide their full name, role, and relevant details from the documents
        
        The question: {message}
        
        Conversation history: {history}
        
        The knowledge: {knowledge}
        
        Answer the question based on the knowledge provided above:
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
