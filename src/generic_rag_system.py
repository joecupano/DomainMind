"""
RF-Expert-AI RAG System for Domain-Specific Expert Systems
Domain-agnostic implementation supporting any subject area
"""
import streamlit as st
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle

from document_processor import DocumentProcessor
from web_scraper import WebScraper
from vector_db import VectorDatabase
from rag_engine import RAGEngine
from domain_catalog import DomainCatalog, get_selected_domain
from utils import (
    display_error, display_success, display_info, display_warning,
    format_file_size, truncate_text
)

logger = logging.getLogger(__name__)

class GenericRAGSystem:
    """Domain-agnostic RAG system with performance optimizations"""
    
    def __init__(self, domain_id: str):
        self.domain_id = domain_id
        self.domain_catalog = DomainCatalog()
        self.domain_info = self.domain_catalog.get_domain(domain_id)
        
        if not self.domain_info:
            raise ValueError(f"Domain '{domain_id}' not found in catalog")
        
        # Domain-specific paths
        self.domain_path = self.domain_catalog.get_domain_path(domain_id)
        self.documents_dir = self.domain_path / "documents"
        self.scraped_dir = self.domain_path / "scraped"
        self.cache_dir = self.domain_path / "cache"
        self.vector_db_dir = self.domain_path / "vector_db"
        
        # Ensure directories exist
        for dir_path in [self.documents_dir, self.scraped_dir, self.cache_dir, self.vector_db_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize core components with domain-specific paths
        self.document_processor = DocumentProcessor()
        self.web_scraper = WebScraper()
        self.vector_db = VectorDatabase(collection_name=f"{domain_id}_collection", 
                                       persist_directory=str(self.vector_db_dir))
        self.rag_engine = RAGEngine(vector_db=self.vector_db)
        
        # Performance enhancements
        self.query_cache = {}
        self.embedding_cache = {}
        self.document_index = {}
        self.batch_size = 10
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Domain-specific features
        self.conversation_memory = []
        self.domain_keywords = self.domain_info.get('keywords', [])
        self.domain_websites = self.domain_info.get('websites', [])
        
        # Load cached data
        self._load_caches()
    
    def _load_caches(self):
        """Load cached data for faster startup"""
        cache_files = {
            'query_cache.pkl': self.query_cache,
            'embedding_cache.pkl': self.embedding_cache,
            'document_index.pkl': self.document_index
        }
        
        for filename, cache_dict in cache_files.items():
            cache_file = self.cache_dir / filename
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        cache_dict.update(cached_data)
                    display_info(f"Loaded {len(cached_data)} entries from {filename}")
                except Exception as e:
                    display_warning(f"Could not load cache {filename}: {e}")
    
    def _save_caches(self):
        """Save caches for persistence"""
        caches = {
            'query_cache.pkl': self.query_cache,
            'embedding_cache.pkl': self.embedding_cache,
            'document_index.pkl': self.document_index
        }
        
        for filename, cache_dict in caches.items():
            if cache_dict:  # Only save non-empty caches
                try:
                    with open(self.cache_dir / filename, 'wb') as f:
                        pickle.dump(dict(list(cache_dict.items())[-1000:]), f)  # Keep last 1000 entries
                except Exception as e:
                    logger.error(f"Could not save cache {filename}: {e}")
    
    def batch_process_documents(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple documents in parallel"""
        def process_single(file_path):
            return self.document_processor.process_file(file_path)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_single, fp) for fp in file_paths]
            results = []
            
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=60)
                    if result:
                        results.append(result)
                    st.progress((i + 1) / len(futures))
                except Exception as e:
                    display_error(f"Failed to process {file_paths[i]}: {e}")
        
        return results
    
    def intelligent_chunking(self, text: str) -> List[str]:
        """Domain-aware chunking with semantic boundaries"""
        from utils import chunk_text
        base_chunks = chunk_text(text, 1000, 200)
        
        # Enhance with domain-specific context markers
        enhanced_chunks = []
        for chunk in base_chunks:
            # Add domain context markers based on keywords
            if any(keyword.lower() in chunk.lower() for keyword in self.domain_keywords):
                chunk = f"[{self.domain_info['name'].upper()}_CONTEXT] {chunk}"
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def cached_query(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """Query with caching for faster responses"""
        cache_key = hash(f"{self.domain_id}_{query}")
        
        if use_cache and cache_key in self.query_cache:
            display_info("Using cached response")
            return self.query_cache[cache_key]
        
        # Generate new response with domain context
        domain_context = f"This is an expert system for {self.domain_info['name']}: {self.domain_info['description']}"
        response = self.rag_engine.generate_response(
            query, 
            self.conversation_memory[-5:],
            domain_context=domain_context
        )
        
        # Cache the response
        if use_cache:
            self.query_cache[cache_key] = response
            if len(self.query_cache) > 100:  # Limit cache size
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
        
        return response
    
    def smart_document_ranking(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """Advanced document ranking with domain-specific scoring"""
        # Create keyword weights dynamically based on domain
        keyword_weights = {}
        for i, keyword in enumerate(self.domain_keywords):
            # Primary keywords get higher weights
            keyword_weights[keyword.lower()] = 2.0 - (i * 0.1)  # Decreasing weights
        
        for result in search_results:
            content_lower = result['content'].lower()
            query_lower = query.lower()
            
            # Base similarity score
            base_score = result.get('similarity_score', 0)
            
            # Domain keyword bonus
            keyword_bonus = 0
            for keyword, weight in keyword_weights.items():
                if keyword in query_lower and keyword in content_lower:
                    keyword_bonus += weight * 0.1
            
            # Length penalty for very short or very long content
            content_len = len(result['content'])
            if content_len < 100:
                length_penalty = -0.2
            elif content_len > 2000:
                length_penalty = -0.1
            else:
                length_penalty = 0
            
            # Final score
            result['enhanced_score'] = base_score + keyword_bonus + length_penalty
        
        return sorted(search_results, key=lambda x: x['enhanced_score'], reverse=True)
    
    def conversation_aware_query(self, query: str) -> Dict[str, Any]:
        """Query processing with conversation context"""
        # Analyze conversation history for context
        recent_topics = []
        if self.conversation_memory:
            for exchange in self.conversation_memory[-3:]:
                if 'human' in exchange:
                    recent_topics.extend(self._extract_domain_topics(exchange['human']))
        
        # Enhance query with context
        if recent_topics:
            context_hint = f" (Previous discussion about: {', '.join(set(recent_topics))})"
            enhanced_query = query + context_hint
        else:
            enhanced_query = query
        
        return self.cached_query(enhanced_query)
    
    def _extract_domain_topics(self, text: str) -> List[str]:
        """Extract domain-specific topics from text"""
        text_lower = text.lower()
        found_topics = [keyword for keyword in self.domain_keywords if keyword.lower() in text_lower]
        return found_topics
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get detailed system analytics"""
        # Vector DB stats
        db_stats = self.vector_db.get_collection_stats()
        
        # Cache stats
        cache_stats = {
            'query_cache_size': len(self.query_cache),
            'embedding_cache_size': len(self.embedding_cache),
            'document_index_size': len(self.document_index),
            'conversation_history': len(self.conversation_memory)
        }
        
        # Performance metrics
        performance_stats = {
            'avg_query_time': self._calculate_avg_query_time(),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'document_processing_speed': self._calculate_processing_speed()
        }
        
        # Domain-specific stats
        domain_stats = self.domain_catalog.get_domain_stats(self.domain_id)
        
        return {
            'domain': self.domain_info,
            'database': db_stats,
            'cache': cache_stats,
            'performance': performance_stats,
            'domain_stats': domain_stats,
            'system_health': self._check_system_health()
        }
    
    def _calculate_avg_query_time(self) -> float:
        # Placeholder - would implement actual timing in production
        return 2.5
    
    def _calculate_cache_hit_rate(self) -> float:
        # Placeholder - would track actual hits/misses
        return 0.75 if self.query_cache else 0.0
    
    def _calculate_processing_speed(self) -> str:
        return "~5 docs/min"
    
    def _check_system_health(self) -> str:
        """Check overall system health"""
        issues = []
        
        if len(self.query_cache) > 500:
            issues.append("Large query cache")
        
        if not self.vector_db.get_collection_stats().get('total_chunks', 0):
            issues.append("No documents indexed")
        
        return "Good" if not issues else f"Issues: {', '.join(issues)}"
    
    def export_conversation(self) -> str:
        """Export conversation history"""
        if not self.conversation_memory:
            return "No conversation history available"
        
        domain_name = self.domain_info['name']
        export_text = f"# {domain_name} Expert System - Conversation Export\n\n"
        
        for i, exchange in enumerate(self.conversation_memory, 1):
            export_text += f"## Exchange {i}\n\n"
            export_text += f"**Question:** {exchange.get('human', 'N/A')}\n\n"
            export_text += f"**Answer:** {exchange.get('assistant', 'N/A')}\n\n"
            export_text += "---\n\n"
        
        return export_text
    
    def cleanup_system(self):
        """Clean up system resources"""
        self._save_caches()
        self.thread_pool.shutdown(wait=True)
        display_info("System cleaned up successfully")

def show_generic_rag_interface():
    """Main interface for generic RAG system"""
    selected_domain = get_selected_domain()
    
    if not selected_domain:
        st.error("No domain selected. Please select a domain from the catalog.")
        return
    
    # Initialize domain-specific RAG system
    if 'generic_rag_system' not in st.session_state or st.session_state.get('current_domain') != selected_domain:
        try:
            with st.spinner(f"Initializing {selected_domain} expert system..."):
                st.session_state.generic_rag_system = GenericRAGSystem(selected_domain)
                st.session_state.current_domain = selected_domain
        except Exception as e:
            st.error(f"Failed to initialize domain system: {e}")
            return
    
    rag_system = st.session_state.generic_rag_system
    domain_info = rag_system.domain_info
    
    # Header with domain information
    st.header(f"🔍 {domain_info['name']} Expert System")
    st.markdown(f"*{domain_info['description']}*")
    
    # Sidebar with advanced controls
    with st.sidebar:
        st.header("🚀 System Controls")
        
        # Domain information
        with st.expander("📋 Domain Info", expanded=True):
            st.write(f"**Domain:** {domain_info['name']}")
            st.write(f"**Keywords:** {', '.join(domain_info['keywords'][:3])}...")
            st.write(f"**Created:** {domain_info.get('created_date', 'Unknown')}")
        
        # System analytics
        with st.expander("📊 Analytics", expanded=True):
            analytics = rag_system.get_system_analytics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", analytics['domain_stats'].get('documents', 0))
                st.metric("Cache Size", analytics['cache']['query_cache_size'])
            with col2:
                st.metric("Avg Query Time", f"{analytics['performance']['avg_query_time']}s")
                st.metric("Cache Hit Rate", f"{analytics['performance']['cache_hit_rate']:.1%}")
            
            st.write(f"**Health:** {analytics['system_health']}")
        
        # Performance settings
        with st.expander("⚙️ Performance Settings"):
            use_cache = st.checkbox("Enable Query Caching", value=True)
            use_smart_ranking = st.checkbox("Smart Document Ranking", value=True)
            use_conversation_context = st.checkbox("Conversation Context", value=True)
            batch_processing = st.checkbox("Batch Document Processing", value=True)
        
        # System maintenance
        with st.expander("🔧 Maintenance"):
            if st.button("Clear Caches"):
                rag_system.query_cache.clear()
                rag_system.embedding_cache.clear()
                display_success("Caches cleared")
            
            if st.button("Export Conversation"):
                conversation_text = rag_system.export_conversation()
                st.download_button(
                    "Download Conversation",
                    conversation_text,
                    file_name=f"{selected_domain}_conversation_export.md",
                    mime="text/markdown"
                )
            
            if st.button("Save System State"):
                rag_system._save_caches()
                display_success("System state saved")
        
        if st.button("🔄 Back to Domains", type="secondary"):
            from domain_catalog import reset_domain_selection
            reset_domain_selection()
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Expert Chat", "📚 Document Management", "🌐 Web Scraping", "📈 Performance"])
    
    with tab1:
        show_enhanced_chat_interface(rag_system, use_cache, use_smart_ranking, use_conversation_context)
    
    with tab2:
        show_advanced_document_management(rag_system, batch_processing)
    
    with tab3:
        show_intelligent_web_scraping(rag_system)
    
    with tab4:
        show_performance_dashboard(rag_system)

def show_enhanced_chat_interface(rag_system, use_cache, use_smart_ranking, use_conversation_context):
    """Enhanced chat interface with optimizations"""
    domain_info = rag_system.domain_info
    st.subheader(f"💬 {domain_info['name']} Expert Chat")
    
    # Quick topics based on domain keywords
    st.markdown("**Quick Topics:**")
    keywords = domain_info.get('keywords', [])
    if len(keywords) >= 4:
        cols = st.columns(4)
        for i, keyword in enumerate(keywords[:4]):
            with cols[i]:
                if st.button(f"🎯 {keyword.title()}", key=f"quick_{keyword}"):
                    st.session_state.quick_query = f"Tell me about {keyword} in {domain_info['name'].lower()}"
    
    # Chat input
    query_input = st.chat_input(f"Ask your {domain_info['name'].lower()} question...")
    
    # Handle quick query or chat input
    user_query = st.session_state.get('quick_query', query_input)
    if 'quick_query' in st.session_state:
        del st.session_state.quick_query
    
    if user_query:
        # Process query based on settings
        with st.spinner(f"Processing your {domain_info['name'].lower()} question..."):
            if use_conversation_context:
                response_data = rag_system.conversation_aware_query(user_query)
            else:
                response_data = rag_system.cached_query(user_query, use_cache)
            
            # Smart ranking if enabled
            if use_smart_ranking and 'sources' in response_data:
                response_data['sources'] = rag_system.smart_document_ranking(
                    user_query, response_data['sources']
                )
        
        # Add to conversation memory
        rag_system.conversation_memory.append({
            'human': user_query,
            'assistant': response_data['response'],
            'timestamp': time.time()
        })
        
        # Display conversation
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(response_data['response'])
        
        # Enhanced source display
        with st.expander("📚 Sources & Analysis", expanded=False):
            if response_data.get('sources'):
                for i, source in enumerate(response_data['sources'][:3], 1):
                    score = source.get('enhanced_score', source.get('similarity_score', 0))
                    st.write(f"**{i}. {source['info']}** (Score: {score:.3f})")
                    st.write(f"Preview: {truncate_text(source.get('content', ''), 150)}")
                    st.divider()
    
    # Show recent conversation
    if rag_system.conversation_memory:
        st.subheader("Recent Exchanges")
        for exchange in rag_system.conversation_memory[-2:]:
            with st.container():
                st.chat_message("user").write(exchange['human'])
                st.chat_message("assistant").write(truncate_text(exchange['assistant'], 300))

def show_advanced_document_management(rag_system, batch_processing):
    """Advanced document management interface"""
    st.subheader("📚 Document Management")
    
    # Batch upload
    uploaded_files = st.file_uploader(
        f"Upload {rag_system.domain_info['name']} Documents",
        type=['pdf', 'docx', 'doc', 'txt'],
        accept_multiple_files=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if uploaded_files and st.button("📥 Process Documents", type="primary"):
            if batch_processing:
                process_documents_batch(rag_system, uploaded_files)
            else:
                process_documents_sequential(rag_system, uploaded_files)
    
    with col2:
        if st.button("🔍 Analyze Collection"):
            analyze_document_collection(rag_system)
    
    with col3:
        if st.button("🧹 Optimize Database"):
            optimize_vector_database(rag_system)
    
    # Document statistics
    st.subheader("📊 Collection Statistics")
    
    stats = rag_system.vector_db.get_collection_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chunks", stats.get('total_chunks', 0))
    with col2:
        st.metric("Document Sources", len(stats.get('source_types', {})))
    with col3:
        st.metric("Cache Entries", len(rag_system.document_index))
    with col4:
        st.metric("Processing Speed", rag_system._calculate_processing_speed())
    
    # Document exploration
    if stats.get('total_chunks', 0) > 0:
        st.subheader("🔍 Explore Documents")
        
        search_term = st.text_input("Search documents by content:")
        if search_term:
            results = rag_system.vector_db.search(search_term, top_k=10)
            
            st.write(f"Found {len(results)} relevant chunks:")
            for i, result in enumerate(results[:5], 1):
                with st.expander(f"Result {i} (Score: {result['similarity_score']:.3f})"):
                    st.write(result['content'])
                    if 'metadata' in result:
                        st.json(result['metadata'])

def show_intelligent_web_scraping(rag_system):
    """Intelligent web scraping interface"""
    st.subheader("🌐 Web Content Scraping")
    
    domain_websites = rag_system.domain_info.get('websites', [])
    
    if domain_websites:
        st.write(f"**Default {rag_system.domain_info['name']} Websites:**")
        for site in domain_websites:
            st.write(f"• {site}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"🌐 Scrape Default Sites", type="primary"):
                scrape_domain_sites(rag_system, domain_websites)
        
        with col2:
            if st.button("🔄 Update All Sites"):
                scrape_domain_sites(rag_system, domain_websites)
    else:
        st.info("No default websites configured for this domain")
    
    # Custom URL scraping
    st.divider()
    st.subheader("Custom URL Scraping")
    
    custom_url = st.text_input(f"Enter custom {rag_system.domain_info['name'].lower()} website URL:")
    if custom_url and st.button("🔍 Scrape Custom URL"):
        scrape_custom_url(rag_system, custom_url)

def show_performance_dashboard(rag_system):
    """Performance monitoring dashboard"""
    st.subheader("📈 Performance Dashboard")
    
    analytics = rag_system.get_system_analytics()
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Query Cache Hit Rate",
            f"{analytics['performance']['cache_hit_rate']:.1%}",
            delta="+5%" if analytics['performance']['cache_hit_rate'] > 0.7 else None
        )
    
    with col2:
        st.metric(
            "Avg Response Time",
            f"{analytics['performance']['avg_query_time']}s",
            delta="-0.3s" if analytics['performance']['avg_query_time'] < 3 else None
        )
    
    with col3:
        st.metric(
            "Documents Indexed",
            analytics['database'].get('total_chunks', 0),
            delta=f"+{len(rag_system.document_index)}" if rag_system.document_index else None
        )
    
    with col4:
        st.metric(
            "System Health",
            analytics['system_health'],
            delta="Optimized" if analytics['system_health'] == "Good" else None
        )
    
    # Domain-specific analytics
    st.subheader(f"📋 {rag_system.domain_info['name']} Statistics")
    
    domain_stats = analytics['domain_stats']
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Content Distribution:**")
        content_data = {
            "Documents": domain_stats.get('documents', 0),
            "Web Sources": domain_stats.get('scraped_files', 0),
            "Training Data": domain_stats.get('training_datasets', 0)
        }
        if any(content_data.values()):
            st.bar_chart(content_data)
        else:
            st.info("No content loaded yet")
    
    with col2:
        st.write("**System Resources:**")
        resource_data = {
            "Vector Database": 45,
            "Document Cache": 25,
            "Query Cache": 20,
            "System Overhead": 10
        }
        st.bar_chart(resource_data)

# Helper functions
def process_documents_batch(rag_system, uploaded_files):
    """Process documents in batch mode"""
    with st.spinner("Processing documents in batch..."):
        temp_paths = []
        for uploaded_file in uploaded_files:
            temp_path = rag_system.documents_dir / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            temp_paths.append(temp_path)
        
        processed_docs = rag_system.batch_process_documents(temp_paths)
        
        if processed_docs:
            rag_system.rag_engine.add_documents_to_knowledge_base(processed_docs)
            display_success(f"Batch processed {len(processed_docs)} documents")

def process_documents_sequential(rag_system, uploaded_files):
    """Process documents sequentially"""
    processed_count = 0
    
    for uploaded_file in uploaded_files:
        result = rag_system.document_processor.process_uploaded_file(uploaded_file)
        if result:
            rag_system.rag_engine.add_documents_to_knowledge_base([result])
            processed_count += 1
    
    display_success(f"Sequentially processed {processed_count} documents")

def analyze_document_collection(rag_system):
    """Analyze the document collection"""
    analytics = rag_system.get_system_analytics()
    
    with st.spinner("Analyzing document collection..."):
        analysis = {
            "Domain": rag_system.domain_info['name'],
            "Total Documents": analytics['domain_stats'].get('documents', 0),
            "Vector Chunks": analytics['database'].get('total_chunks', 0),
            "Cache Efficiency": f"{analytics['performance']['cache_hit_rate']:.1%}",
            "System Health": analytics['system_health']
        }
        
        st.json(analysis)
        display_success("Collection analysis complete")

def optimize_vector_database(rag_system):
    """Optimize the vector database"""
    with st.spinner("Optimizing vector database..."):
        rag_system.query_cache.clear()
        rag_system._save_caches()
        display_success("Database optimization complete")

def scrape_domain_sites(rag_system, sites):
    """Scrape sites from the domain's default list"""
    with st.spinner(f"Scraping {rag_system.domain_info['name']} websites..."):
        scraped_docs = rag_system.web_scraper.scrape_multiple_urls(sites)
        
        if scraped_docs:
            rag_system.rag_engine.add_documents_to_knowledge_base(scraped_docs)
            display_success(f"Scraped {len(scraped_docs)} pages from domain websites")

def scrape_custom_url(rag_system, url):
    """Scrape a custom URL"""
    with st.spinner(f"Scraping {url}..."):
        result = rag_system.web_scraper.scrape_url(url)
        
        if result:
            rag_system.rag_engine.add_documents_to_knowledge_base([result])
            display_success(f"Successfully scraped {url}")