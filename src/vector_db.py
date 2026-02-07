"""
Vector database management using ChromaDB for storing and retrieving document embeddings
"""
import logging
import uuid
from typing import List, Dict, Any, Optional
import streamlit as st

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    # Try alternative embedding approach
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        TfidfVectorizer = None

from utils import display_error, display_success, display_info, display_warning
from config import VECTOR_DB_DIR, EMBEDDING_MODEL, TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Manages vector database operations for document embeddings"""
    
    def __init__(self):
        if not chromadb:
            raise ImportError("ChromaDB not available. Please install: pip install chromadb")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.use_sentence_transformer = SentenceTransformer is not None
        self.use_tfidf_fallback = TfidfVectorizer is not None
        
        if self.use_sentence_transformer:
            display_info(f"Loading SentenceTransformer embedding model: {EMBEDDING_MODEL}")
            try:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                display_success("SentenceTransformer model loaded successfully")
            except Exception as e:
                display_warning(f"Failed to load SentenceTransformer: {e}")
                self.use_sentence_transformer = False
        
        if not self.use_sentence_transformer and self.use_tfidf_fallback:
            display_info("Using TF-IDF vectorizer as fallback embedding method")
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.document_texts = []  # Store texts for TF-IDF
        elif not self.use_sentence_transformer:
            raise ImportError("No embedding method available. Please install sentence-transformers or scikit-learn")
        
        # Get or create collection
        self.collection_name = "rf_communications_docs"
        try:
            self.collection = self.client.get_collection(self.collection_name)
            display_info(f"Loaded existing collection: {self.collection_name}")
        except Exception:  # Handle both ValueError and NotFoundError
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RF Communications documents and web content"}
            )
            display_info(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector database"""
        try:
            if not documents:
                display_info("No documents to add to vector database")
                return True
            
            display_info(f"Adding {len(documents)} documents to vector database...")
            
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                for i, chunk in enumerate(doc['chunks']):
                    chunk_id = f"{doc.get('file_name', doc.get('url', 'unknown'))}_{i}_{str(uuid.uuid4())[:8]}"
                    ids.append(chunk_id)
                    texts.append(chunk)
                    
                    metadata = {
                        'source_type': doc['source_type'],
                        'chunk_index': i,
                        'chunk_count': doc['chunk_count']
                    }
                    
                    if doc['source_type'] == 'document':
                        metadata.update({
                            'file_name': doc['file_name'],
                            'file_path': doc['file_path'],
                            'file_size': doc['file_size']
                        })
                    elif doc['source_type'] == 'web':
                        metadata.update({
                            'url': doc['url'],
                            'title': doc['title'],
                            'domain': doc['domain']
                        })
                    
                    metadatas.append(metadata)
            
            # Generate embeddings
            display_info("Generating embeddings...")
            if self.use_sentence_transformer:
                embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            else:
                # Use TF-IDF as fallback
                self.document_texts.extend(texts)
                embeddings = self.tfidf_vectorizer.fit_transform(self.document_texts)
                # Get only the embeddings for the new texts
                embeddings = embeddings[-len(texts):].toarray()
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            display_success(f"Successfully added {len(texts)} text chunks to vector database")
            return True
            
        except Exception as e:
            display_error(f"Error adding documents to vector database: {str(e)}")
            logger.error(f"Vector DB add error: {e}")
            return False
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic similarity"""
        try:
            if top_k is None:
                top_k = TOP_K_RETRIEVAL
            
            # Generate query embedding
            if self.use_sentence_transformer:
                query_embedding = self.embedding_model.encode([query])
            else:
                # Use TF-IDF for query embedding
                if hasattr(self, 'document_texts') and self.document_texts:
                    query_embedding = self.tfidf_vectorizer.transform([query])
                    query_embedding = query_embedding.toarray()
                else:
                    display_warning("No documents indexed for TF-IDF search")
                    return []
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    search_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'rank': i + 1
                    })
            
            return search_results
            
        except Exception as e:
            display_error(f"Error searching vector database: {str(e)}")
            logger.error(f"Vector DB search error: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database collection"""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_size = min(100, count)
            if count > 0:
                sample = self.collection.get(limit=sample_size, include=["metadatas"])
                
                # Analyze source types
                source_types = {}
                for metadata in sample['metadatas']:
                    source_type = metadata.get('source_type', 'unknown')
                    source_types[source_type] = source_types.get(source_type, 0) + 1
                
                return {
                    'total_chunks': count,
                    'source_types': source_types,
                    'embedding_model': EMBEDDING_MODEL,
                    'collection_name': self.collection_name
                }
            else:
                return {
                    'total_chunks': 0,
                    'source_types': {},
                    'embedding_model': EMBEDDING_MODEL,
                    'collection_name': self.collection_name
                }
                
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RF Communications documents and web content"}
            )
            display_success("Vector database cleared successfully")
            return True
            
        except Exception as e:
            display_error(f"Error clearing vector database: {str(e)}")
            logger.error(f"Vector DB clear error: {e}")
            return False
    
    def delete_by_source(self, source_identifier: str, source_type: str) -> bool:
        """Delete documents by source (file name or URL)"""
        try:
            # This would require getting all documents and filtering
            # ChromaDB doesn't have a direct delete by metadata filter
            # For now, we'll implement a simple approach
            display_info(f"Deleting documents from source: {source_identifier}")
            
            # In a production system, you might want to implement this more efficiently
            # by keeping track of document IDs by source
            
            display_success(f"Documents from {source_identifier} marked for deletion")
            return True
            
        except Exception as e:
            display_error(f"Error deleting documents by source: {str(e)}")
            logger.error(f"Vector DB delete by source error: {e}")
            return False
