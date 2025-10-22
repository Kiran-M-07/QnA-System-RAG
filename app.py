import streamlit as st
from rag_helper import (
    EmbeddingManager, 
    VectorStore, 
    RAGRetriever,
    rag_simple,
    rag_advanced
)
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# print(GROQ_API_KEY)

# llm = ChatGroq(
#         api_key=GROQ_API_KEY,
#         model="openai/gpt-oss-20b",  # or you can use "llama2-70b-4096"
#         temperature=0.7
#     )

# print(llm)

# Set page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for components
if 'retriever' not in st.session_state:
    # Initialize components
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    st.session_state.retriever = RAGRetriever(vector_store, embedding_manager)
    st.session_state.llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="openai/gpt-oss-20b",  # or you can use "llama2-70b-4096"
        temperature=0.7
    )

def process_query(query: str, use_advanced: bool = False):
    """Process the user query using either simple or advanced RAG"""
    try:
        if use_advanced:
            response = rag_advanced(
                query=query,
                retriever=st.session_state.retriever,
                llm=st.session_state.llm,
                return_context=True
            )
            if isinstance(response, str):  # Handle error message
                return {"error": response}
            return response
        else:
            response = rag_simple(
                query=query,
                retriever=st.session_state.retriever,
                llm=st.session_state.llm
            )
            if isinstance(response, str):  # Handle both normal responses and error messages
                return {"answer": response}
    except Exception as e:
        return {"error": str(e)}

# Main app interface
st.title("📚 Document Q&A System")
st.write("Ask questions about your documents and get AI-powered answers!")

# Query input
query = st.text_area("Enter your question:", height=100)

# Advanced options
with st.expander("Advanced Options"):
    use_advanced = st.checkbox("Use advanced RAG (shows sources and confidence)")
    top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=3)
    min_confidence = st.slider("Minimum confidence score", min_value=0.0, max_value=1.0, value=0.2)

# Submit button
if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question!")
    else:
        with st.spinner("Processing your question..."):
            response = process_query(query, use_advanced)
            
            if "error" in response:
                st.error(f"An error occurred: {response['error']}")
            else:
                # Display answer
                st.markdown("### Answer")
                st.write(response["answer"])
                
                # If using advanced RAG, show additional information
                if use_advanced and "confidence" in response:
                    st.markdown("### Confidence Score")
                    col1, col2 = st.columns([0.7, 0.3])
                    with col1:
                        st.progress(response["confidence"])
                    with col2:
                        st.markdown(f"**{response['confidence']:.2%}**")
                    
                    st.markdown("### Sources")
                    for idx, source in enumerate(response["sources"], 1):
                        with st.expander(f"Source {idx} - {source['source']} (Score: {source['score']:.2f})"):
                            st.write(f"Page: {source['page']}")
                            st.write("Preview:")
                            st.write(source['preview'])
                            
# Sidebar with information
with st.sidebar:
    st.markdown("### About")
    st.write("""
    This Q&A system uses RAG (Retrieval-Augmented Generation) to provide accurate answers based on your documents.
    
    Features:
    - Simple and advanced query modes
    - Source tracking
    - Confidence scores
    - Document previews
    """)
    
    st.markdown("### Tips")
    st.write("""
    - Be specific in your questions
    - Use advanced mode to see sources
    - Adjust confidence threshold for better results
    """)