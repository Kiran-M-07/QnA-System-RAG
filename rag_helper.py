import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import List, Dict, Any
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

class EmbeddingManager : 
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Model {self.model_name} loaded successfully. Embdedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise e
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded. Call _load_model() first.")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings


## Vector Store using ChromaDB
class VectorStore:
    
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
        
    def _initialize_store(self):
        
        try:   
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF Document Embdeddings for RAG"})
            
            print(f"Vector store initialized at {self.persist_directory} with collection {self.collection_name}")
            print(f"Existing documents in collectio : {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise e
        
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents and embeddings must match.")
        
        print(f"Adding {len(documents)} documents to the vector store...")
        
        # prepare data for chromadb
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
            # generate unique id
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # prepare metadata
            metadata = dict(doc.metadata) if doc.metadata else {}
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            
            # document content
            documents_text.append(doc.page_content)
            
            # embedding
            embeddings_list.append(embedding.tolist())
            
        # add to collection
        try :
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_text,
                embeddings=embeddings_list
            )
            print(f"Documents added successfully. Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise e    
        
## Retriever module 
class RAGRetriever:
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        
        
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str,Any]]:
        # generate embedding for the query
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # perform similarity search in vector store 
        try : 
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
        
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance  # convert distance to similarity score
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "document": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            'distance': distance,
                            'rank': i+1
                        })
                        
                print(f"Retrieved {len(retrieved_docs)} documents for the query: '{query}'")
            else:
                print(f"No documents retrieved for the query: '{query}'")
                
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
            
def process_all_pdfs(pdf_directory):
    """Process all PDF files in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

def rag_simple(query: str, retriever: RAGRetriever, llm: ChatGroq, top_k: int = 3) -> str:
    """Simple RAG pipeline that retrieves documents and generates a response"""
    # retrieve relevant documents
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['document'] for doc in results]) if results else "No relevant documents found."
    if not context: 
        return "No relevant documents found."
    
    prompt = f"""Use the following context to answer the question conscisely.
    Context: 
    {context}

    Question: {query}

    Answer:""".strip()
    
    response = llm.invoke([prompt.format(context=context, query=query)])
    return response.content

def rag_advanced(query: str, retriever: RAGRetriever, llm: ChatGroq, top_k: int = 5, 
                min_score: float = 0.2, return_context: bool = False) -> Dict[str, Any]:
    """
    Advanced RAG pipeline with additional features like confidence scores and source tracking
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return "No relevant documents found."   
    
    context = "\n\n".join([doc['document'] for doc in results])
    
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', doc['metadata'].get('pages', 'unknown')),
        'score': doc['similarity_score'],
        'preview': doc['document'][:300] + "..." if len(doc['document']) > 300 else doc['document']
    } for doc in results]
    
    confidence = max(doc['similarity_score'] for doc in results)
    
    prompt = f"""Use the following context to answer the question conscisely.
    Context: 
    {context}
    Question: {query}
    Answer:""".strip()
    
    response = llm.invoke([prompt.format(context=context, query=query)])
    
    output = {
        "answer": response.content,
        "confidence": confidence,
        "sources": sources
    }
    if return_context:
        output['context'] = context
        
    return output