"""
RAG (Retrieval-Augmented Generation) engine for RF communications expert system
"""
import logging
import json
from typing import List, Dict, Any, Optional
import streamlit as st

try:
    import requests
except ImportError:
    requests = None

from vector_db import VectorDatabase
from utils import display_error, display_success, display_info, display_warning
from config import RF_EXPERT_PROMPT, OLLAMA_HOST, LLAMA_MODEL, TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

class RAGEngine:
    """RAG engine that combines retrieval and generation for RF communications queries"""
    
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.llama_available = self._check_llama_availability()
        
    def _check_llama_availability(self) -> bool:
        """Check if Ollama/Llama is available"""
        try:
            if not requests:
                display_warning("Requests library not available")
                return False
            
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                
                if any(LLAMA_MODEL in model for model in available_models):
                    display_success(f"Llama model '{LLAMA_MODEL}' is available")
                    return True
                else:
                    display_warning(f"Llama model '{LLAMA_MODEL}' not found. Available models: {available_models}")
                    return False
            else:
                display_warning(f"Ollama server not responding. Status code: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            display_warning(f"Cannot connect to Ollama server at {OLLAMA_HOST}: {str(e)}")
            return False
        except Exception as e:
            display_warning(f"Error checking Llama availability: {str(e)}")
            return False
    
    def generate_response(self, query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate a response using RAG approach"""
        try:
            # Step 1: Retrieve relevant context
            display_info("Searching for relevant context...")
            search_results = self.vector_db.search(query, TOP_K_RETRIEVAL)
            
            if not search_results:
                return {
                    'response': "I don't have relevant information in my knowledge base to answer your question about RF communications. Please ensure documents have been uploaded and processed.",
                    'sources': [],
                    'context_used': "",
                    'error': "No relevant context found"
                }
            
            # Filter results by similarity threshold
            relevant_results = [
                result for result in search_results 
                if result['similarity_score'] >= SIMILARITY_THRESHOLD
            ]
            
            if not relevant_results:
                return {
                    'response': f"I found some potentially related information, but it doesn't seem directly relevant to your question (similarity below {SIMILARITY_THRESHOLD}). Could you rephrase your question or provide more specific details?",
                    'sources': [],
                    'context_used': "",
                    'error': "No sufficiently relevant context found"
                }
            
            # Step 2: Prepare context
            context_parts = []
            sources = []
            
            for result in relevant_results[:TOP_K_RETRIEVAL]:
                context_parts.append(result['content'])
                
                # Prepare source information
                metadata = result['metadata']
                if metadata['source_type'] == 'document':
                    source_info = f"Document: {metadata['file_name']}"
                elif metadata['source_type'] == 'web':
                    source_info = f"Web: {metadata['title']} ({metadata['domain']})"
                else:
                    source_info = "Unknown source"
                
                sources.append({
                    'type': metadata['source_type'],
                    'info': source_info,
                    'similarity': result['similarity_score'],
                    'rank': result['rank']
                })
            
            context = "\n\n".join(context_parts)
            
            # Step 3: Generate response
            if self.llama_available:
                response_text = self._generate_llama_response(query, context, conversation_history)
            else:
                response_text = self._generate_fallback_response(query, context)
            
            return {
                'response': response_text,
                'sources': sources,
                'context_used': context,
                'retrieval_count': len(search_results),
                'relevant_count': len(relevant_results)
            }
            
        except Exception as e:
            display_error(f"Error generating response: {str(e)}")
            logger.error(f"RAG generation error: {e}")
            return {
                'response': "I encountered an error while processing your question. Please try again.",
                'sources': [],
                'context_used': "",
                'error': str(e)
            }
    
    def _generate_llama_response(self, query: str, context: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate response using Llama model"""
        try:
            # Prepare the prompt
            prompt = RF_EXPERT_PROMPT.format(context=context, question=query)
            
            # Add conversation history if available
            if conversation_history:
                history_text = "\n".join([
                    f"Human: {msg['human']}\nAssistant: {msg['assistant']}"
                    for msg in conversation_history[-3:]  # Last 3 exchanges
                ])
                prompt = f"Previous conversation:\n{history_text}\n\n{prompt}"
            
            # Make request to Ollama
            payload = {
                "model": LLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Sorry, I could not generate a response.')
            else:
                display_error(f"Llama API error: {response.status_code}")
                return self._generate_fallback_response(query, context)
                
        except Exception as e:
            display_error(f"Error calling Llama API: {str(e)}")
            logger.error(f"Llama API error: {e}")
            return self._generate_fallback_response(query, context)
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate fallback response when Llama is not available"""
        return f"""Based on the available RF communications documentation, here's the relevant information I found:

**Context from Knowledge Base:**
{context[:1500]}{'...' if len(context) > 1500 else ''}

**Note:** This is a direct excerpt from the documentation. For more detailed analysis and interpretation, please ensure the Llama model is properly configured and running.

**Your Question:** {query}

Please review the context above for information relevant to your RF communications query."""
    
    def add_documents_to_knowledge_base(self, documents: List[Dict[str, Any]]) -> bool:
        """Add new documents to the knowledge base"""
        return self.vector_db.add_documents(documents)
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        stats = self.vector_db.get_collection_stats()
        stats['llama_available'] = self.llama_available
        stats['llama_model'] = LLAMA_MODEL
        stats['ollama_host'] = OLLAMA_HOST
        return stats
    
    def clear_knowledge_base(self) -> bool:
        """Clear the entire knowledge base"""
        return self.vector_db.clear_collection()
    
    def test_connection(self) -> Dict[str, Any]:
        """Test all system connections"""
        results = {
            'vector_db': True,
            'llama': self.llama_available,
            'embedding_model': True
        }
        
        try:
            # Test vector database
            test_results = self.vector_db.search("test query", top_k=1)
            results['vector_db'] = True
        except Exception as e:
            results['vector_db'] = False
            results['vector_db_error'] = str(e)
        
        try:
            # Test embedding model
            self.vector_db.embedding_model.encode(["test"])
            results['embedding_model'] = True
        except Exception as e:
            results['embedding_model'] = False
            results['embedding_model_error'] = str(e)
        
        return results
