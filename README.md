# Document Q&A System with RAG

A powerful Question-Answering system built using Retrieval-Augmented Generation (RAG) with Groq LLM and ChromaDB. This system allows users to ask questions about their documents and receive accurate, context-aware responses.

## 🚀 Features

- **Smart Document Understanding**: Process and understand PDF documents efficiently
- **Advanced RAG Implementation**: Combines the power of retrieval and generation for accurate answers
- **Real-time Confidence Scoring**: See how confident the system is about each answer
- **Source Tracking**: View the exact sources used to generate answers
- **User-friendly Interface**: Built with Streamlit for easy interaction
- **Document Preview**: See relevant excerpts from source documents

## 🔧 Technology Stack

- **[Groq LLM](https://console.groq.com/)**: Ultra-fast LLM inference
- **[ChromaDB](https://www.trychroma.com/)**: Vector database for efficient document storage and retrieval
- **Sentence Transformers**: For generating document embeddings
- **Streamlit**: For the web interface
- **LangChain**: For RAG pipeline orchestration

## ⚡ Why Groq?

Groq offers several advantages for this RAG implementation:

1. **Exceptional Speed**: 
   - Up to 100x faster inference than traditional GPU-based solutions
   - Sub-second response times for complex queries

2. **High-Quality Models**:
   - Access to top models like Mixtral 8x7B and Llama2
   - Consistent and reliable outputs

3. **Cost-Effective**:
   - Pay only for what you use
   - Competitive pricing compared to other LLM providers

## 💾 Why ChromaDB?

ChromaDB is an excellent choice for RAG applications because:

1. **Ease of Use**:
   - Simple, intuitive API
   - Native Python integration
   - Easy setup and maintenance

2. **Performance**:
   - Fast similarity search
   - Efficient vector storage
   - Excellent scaling capabilities

3. **Features**:
   - Multiple embedding function support
   - Rich metadata filtering
   - Persistent storage options

## 🛠️ Setup

1. Create a virtual environment:
   ```bash
   python -m venv rag_env
   source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements_rag.txt
   ```

3. Set up your Groq API key:
   - Sign up at [Groq Console](https://console.groq.com/)
   - Create a `.env` file in the project root
   - Add your API key: `GROQ_API_KEY=your_api_key_here`

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
rag_project/
├── app.py                 # Streamlit web application
├── rag_helper.py         # Core RAG implementation
├── requirements_rag.txt  # Project dependencies
├── .env                 # Environment variables
└── data/
    ├── pdf_files/      # Your PDF documents
    ├── text_files/     # Extracted text
    └── vector_store/   # ChromaDB storage
```

## 🔍 Core Components

1. **EmbeddingManager**: Handles document embedding generation using Sentence Transformers

2. **VectorStore**: Manages document storage and retrieval using ChromaDB

3. **RAGRetriever**: Implements the retrieval logic for finding relevant documents

4. **Processing Functions**: 
   - `process_all_pdfs()`: Batch process PDF documents
   - `rag_simple()`: Basic RAG implementation
   - `rag_advanced()`: Enhanced RAG with confidence scoring and source tracking

## 📚 Usage

1. Place your PDF documents in the `data/pdf_files` directory

2. Launch the application:
   ```bash
   streamlit run app.py
   ```

3. Enter your question in the text area

4. Choose between simple and advanced modes:
   - Simple: Quick answers
   - Advanced: Detailed responses with sources and confidence scores

5. Adjust parameters like:
   - Number of documents to retrieve
   - Minimum confidence threshold
   - Temperature for response generation

## 🤝 Contributing

Feel free to:
- Open issues for bugs or enhancements
- Submit pull requests
- Share feedback and suggestions

## 📝 License

MIT License - feel free to use this project for your own applications!

## 🔗 Additional Resources

- [Groq Documentation](https://console.groq.com/docs/quickstart)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit Documentation](https://docs.streamlit.io/)
