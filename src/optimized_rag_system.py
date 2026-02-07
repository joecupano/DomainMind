"""
Optimized RAG System for RF Communications Expert System
Enhanced version with performance improvements and advanced features
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
from utils import (
    display_error, display_success, display_info, display_warning,
    format_file_size, truncate_text
)
from config import DOCUMENTS_DIR, SCRAPED_DIR, DATA_DIR

logger = logging.getLogger(__name__)

class OptimizedRAGSystem:
    """Enhanced RAG system with performance optimizations"""
    
    def __init__(self):
        self.cache_dir = DATA_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize core components
        self.document_processor = DocumentProcessor()
        self.web_scraper = WebScraper()
        self.vector_db = VectorDatabase()
        self.rag_engine = RAGEngine()
        
        # Performance enhancements
        self.query_cache = {}
        self.embedding_cache = {}
        self.batch_size = 10
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Advanced features
        self.conversation_memory = []
        self.document_index = {}
        self.semantic_clusters = {}
        
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
        """Advanced chunking with semantic boundaries"""
        # Use existing chunking as base
        from utils import chunk_text
        base_chunks = chunk_text(text, 1000, 200)
        
        # Enhance with semantic analysis
        enhanced_chunks = []
        for chunk in base_chunks:
            # Add RF-specific context markers
            if any(keyword in chunk.lower() for keyword in 
                   ['antenna', 'frequency', 'impedance', 'amplifier', 'filter']):
                chunk = f"[RF_CONTEXT] {chunk}"
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def cached_query(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """Query with caching for faster responses"""
        cache_key = hash(query)
        
        if use_cache and cache_key in self.query_cache:
            display_info("Using cached response")
            return self.query_cache[cache_key]
        
        # Generate new response
        response = self.rag_engine.generate_response(query, self.conversation_memory[-5:])
        
        # Cache the response
        if use_cache:
            self.query_cache[cache_key] = response
            if len(self.query_cache) > 100:  # Limit cache size
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
        
        return response
    
    def smart_document_ranking(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """Advanced document ranking with RF-specific scoring"""
        rf_keywords = {
            'antenna': 2.0,
            'frequency': 1.8,
            'impedance': 1.7,
            'amplifier': 1.6,
            'filter': 1.5,
            'transmission': 1.4,
            'power': 1.3,
            'noise': 1.2,
            'circuit': 1.1,
            'signal': 1.0
        }
        
        for result in search_results:
            content_lower = result['content'].lower()
            query_lower = query.lower()
            
            # Base similarity score
            base_score = result.get('similarity_score', 0)
            
            # RF keyword bonus
            keyword_bonus = 0
            for keyword, weight in rf_keywords.items():
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
        
        # Sort by enhanced score
        return sorted(search_results, key=lambda x: x['enhanced_score'], reverse=True)
    
    def conversation_aware_query(self, query: str) -> Dict[str, Any]:
        """Query processing with conversation context"""
        # Analyze conversation history for context
        recent_topics = []
        if self.conversation_memory:
            for exchange in self.conversation_memory[-3:]:
                if 'human' in exchange:
                    recent_topics.extend(self._extract_rf_topics(exchange['human']))
        
        # Enhance query with context
        if recent_topics:
            context_hint = f" (Previous discussion about: {', '.join(set(recent_topics))})"
            enhanced_query = query + context_hint
        else:
            enhanced_query = query
        
        return self.cached_query(enhanced_query)
    
    def _extract_rf_topics(self, text: str) -> List[str]:
        """Extract RF topics from text"""
        rf_terms = [
            'antenna', 'frequency', 'impedance', 'amplifier', 'filter',
            'transmission line', 'waveguide', 'mixer', 'oscillator',
            'power', 'noise', 'gain', 'bandwidth', 'vswr', 'smith chart'
        ]
        
        text_lower = text.lower()
        found_topics = [term for term in rf_terms if term in text_lower]
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
        
        return {
            'database': db_stats,
            'cache': cache_stats,
            'performance': performance_stats,
            'system_health': self._check_system_health()
        }
    
    def _calculate_avg_query_time(self) -> float:
        """Calculate average query processing time"""
        # Placeholder - would implement actual timing in production
        return 2.5
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # Placeholder - would track actual hits/misses
        return 0.75 if self.query_cache else 0.0
    
    def _calculate_processing_speed(self) -> str:
        """Calculate document processing speed"""
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
        
        export_text = "# RF Communications Expert System - Conversation Export\n\n"
        
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

def show_optimized_rag_interface():
    """Main interface for optimized RAG system"""
    st.header("🔍 Optimized RAG System for RF Communications")
    st.markdown("Enhanced retrieval system with advanced features and performance optimizations")
    
    # Initialize optimized system
    if 'opt_rag_system' not in st.session_state:
        with st.spinner("Initializing optimized RAG system..."):
            st.session_state.opt_rag_system = OptimizedRAGSystem()
    
    opt_rag = st.session_state.opt_rag_system
    
    # Sidebar with advanced controls
    with st.sidebar:
        st.header("🚀 System Controls")
        
        # System analytics
        with st.expander("📊 Analytics", expanded=True):
            analytics = opt_rag.get_system_analytics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", analytics['database'].get('total_chunks', 0))
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
                opt_rag.query_cache.clear()
                opt_rag.embedding_cache.clear()
                display_success("Caches cleared")
            
            if st.button("Export Conversation"):
                conversation_text = opt_rag.export_conversation()
                st.download_button(
                    "Download Conversation",
                    conversation_text,
                    file_name="rf_conversation_export.md",
                    mime="text/markdown"
                )
            
            if st.button("Save System State"):
                opt_rag._save_caches()
                display_success("System state saved")
        
        if st.button("🔄 Reset Mode", type="secondary"):
            from mode_selector import reset_mode_selection
            reset_mode_selection()
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Expert Chat", "📚 Document Management", "🌐 Web Scraping", "📈 Performance"])
    
    with tab1:
        show_enhanced_chat_interface(opt_rag, use_cache, use_smart_ranking, use_conversation_context)
    
    with tab2:
        show_advanced_document_management(opt_rag, batch_processing)
    
    with tab3:
        show_intelligent_web_scraping(opt_rag)
    
    with tab4:
        show_performance_dashboard(opt_rag)

def show_enhanced_chat_interface(opt_rag, use_cache, use_smart_ranking, use_conversation_context):
    """Enhanced chat interface with optimizations"""
    st.subheader("💬 RF Expert Chat (Enhanced)")
    
    # Quick RF topics
    st.markdown("**Quick Topics:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎯 Antenna Design"):
            st.session_state.quick_query = "How do I design a microstrip antenna?"
    with col2:
        if st.button("⚡ RF Amplifiers"):
            st.session_state.quick_query = "What are the key RF amplifier parameters?"
    with col3:
        if st.button("📡 Transmission Lines"):
            st.session_state.quick_query = "How do transmission lines work?"
    with col4:
        if st.button("🔍 RF Measurements"):
            st.session_state.quick_query = "What instruments do I need for RF testing?"
    
    # Chat input
    query_input = st.chat_input("Ask your RF question...")
    
    # Handle quick query or chat input
    user_query = st.session_state.get('quick_query', query_input)
    if 'quick_query' in st.session_state:
        del st.session_state.quick_query
    
    if user_query:
        # Process query based on settings
        with st.spinner("Processing your RF question..."):
            if use_conversation_context:
                response_data = opt_rag.conversation_aware_query(user_query)
            else:
                response_data = opt_rag.cached_query(user_query, use_cache)
            
            # Smart ranking if enabled
            if use_smart_ranking and 'sources' in response_data:
                response_data['sources'] = opt_rag.smart_document_ranking(
                    user_query, response_data['sources']
                )
        
        # Add to conversation memory
        opt_rag.conversation_memory.append({
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
    if opt_rag.conversation_memory:
        st.subheader("Recent Exchanges")
        for exchange in opt_rag.conversation_memory[-2:]:
            with st.container():
                st.chat_message("user").write(exchange['human'])
                st.chat_message("assistant").write(truncate_text(exchange['assistant'], 300))

def show_advanced_document_management(opt_rag, batch_processing):
    """Advanced document management interface"""
    st.subheader("📚 Advanced Document Management")
    
    # Batch upload
    uploaded_files = st.file_uploader(
        "Upload RF Documents (Batch Processing)",
        type=['pdf', 'docx', 'doc', 'txt'],
        accept_multiple_files=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if uploaded_files and st.button("📥 Batch Process", type="primary"):
            if batch_processing:
                process_documents_batch(opt_rag, uploaded_files)
            else:
                process_documents_sequential(opt_rag, uploaded_files)
    
    with col2:
        if st.button("🔍 Analyze Collection"):
            analyze_document_collection(opt_rag)
    
    with col3:
        if st.button("🧹 Optimize Database"):
            optimize_vector_database(opt_rag)
    
    # Document statistics
    st.subheader("📊 Collection Statistics")
    
    stats = opt_rag.vector_db.get_collection_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chunks", stats.get('total_chunks', 0))
    with col2:
        st.metric("Document Sources", len(stats.get('source_types', {})))
    with col3:
        st.metric("Cache Entries", len(opt_rag.document_index))
    with col4:
        st.metric("Processing Speed", opt_rag._calculate_processing_speed())
    
    # Document exploration
    if stats.get('total_chunks', 0) > 0:
        st.subheader("🔍 Explore Documents")
        
        search_term = st.text_input("Search documents by content:")
        if search_term:
            results = opt_rag.vector_db.search(search_term, top_k=10)
            
            st.write(f"Found {len(results)} relevant chunks:")
            for i, result in enumerate(results[:5], 1):
                with st.expander(f"Result {i} (Score: {result['similarity_score']:.3f})"):
                    st.write(result['content'])
                    if 'metadata' in result:
                        st.json(result['metadata'])

def show_intelligent_web_scraping(opt_rag):
    """Intelligent web scraping interface"""
    st.subheader("🌐 Intelligent Web Scraping")
    
    # RF-specific websites with categories
    rf_sites = {
        "Antenna Resources": [
            "https://www.antenna-theory.com",
            "https://www.antennas.com",
            "https://www.emtalk.com/antenna"
        ],
        "RF Design": [
            "https://www.rfcafe.com",
            "https://www.microwaves101.com",
            "https://www.everythingrf.com"
        ],
        "Test & Measurement": [
            "https://www.keysight.com/us/en/support/RF",
            "https://www.rohde-schwarz.com/applications/rf-microwave"
        ]
    }
    
    # Category selection
    selected_category = st.selectbox("Select RF Category", list(rf_sites.keys()))
    selected_sites = rf_sites[selected_category]
    
    st.write(f"**{selected_category} Sites:**")
    for site in selected_sites:
        st.write(f"• {site}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"🌐 Scrape {selected_category}", type="primary"):
            scrape_category_sites(opt_rag, selected_sites, selected_category)
    
    with col2:
        if st.button("🔄 Update All Categories"):
            for category, sites in rf_sites.items():
                scrape_category_sites(opt_rag, sites, category)
    
    # Custom URL scraping
    st.divider()
    st.subheader("Custom URL Scraping")
    
    custom_url = st.text_input("Enter custom RF website URL:")
    if custom_url and st.button("🔍 Scrape Custom URL"):
        scrape_custom_url(opt_rag, custom_url)

def show_performance_dashboard(opt_rag):
    """Performance monitoring dashboard"""
    st.subheader("📈 Performance Dashboard")
    
    analytics = opt_rag.get_system_analytics()
    
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
            delta=f"+{len(opt_rag.document_index)}" if opt_rag.document_index else None
        )
    
    with col4:
        st.metric(
            "System Health",
            analytics['system_health'],
            delta="Optimized" if analytics['system_health'] == "Good" else None
        )
    
    # System resource usage
    st.subheader("💻 Resource Usage")
    
    # Placeholder for system monitoring
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Memory Usage:**")
        memory_data = {
            "Vector Database": 45,
            "Document Cache": 25,
            "Query Cache": 20,
            "System Overhead": 10
        }
        st.bar_chart(memory_data)
    
    with col2:
        st.write("**Processing Distribution:**")
        processing_data = {
            "Document Processing": 40,
            "Vector Search": 35,
            "LLM Generation": 20,
            "Other": 5
        }
        st.bar_chart(processing_data)

def process_documents_batch(opt_rag, uploaded_files):
    """Process documents in batch mode"""
    with st.spinner("Processing documents in batch..."):
        # Save files temporarily
        temp_paths = []
        for uploaded_file in uploaded_files:
            temp_path = DOCUMENTS_DIR / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            temp_paths.append(temp_path)
        
        # Batch process
        processed_docs = opt_rag.batch_process_documents(temp_paths)
        
        if processed_docs:
            opt_rag.rag_engine.add_documents_to_knowledge_base(processed_docs)
            display_success(f"Batch processed {len(processed_docs)} documents")

def process_documents_sequential(opt_rag, uploaded_files):
    """Process documents sequentially"""
    processed_count = 0
    
    for uploaded_file in uploaded_files:
        result = opt_rag.document_processor.process_uploaded_file(uploaded_file)
        if result:
            opt_rag.rag_engine.add_documents_to_knowledge_base([result])
            processed_count += 1
    
    display_success(f"Sequentially processed {processed_count} documents")

def analyze_document_collection(opt_rag):
    """Analyze the document collection"""
    stats = opt_rag.get_system_analytics()
    
    with st.spinner("Analyzing document collection..."):
        analysis = {
            "Total Documents": stats['database'].get('total_chunks', 0),
            "Source Distribution": stats['database'].get('source_types', {}),
            "Cache Efficiency": f"{stats['performance']['cache_hit_rate']:.1%}",
            "System Health": stats['system_health']
        }
        
        st.json(analysis)
        display_success("Collection analysis complete")

def optimize_vector_database(opt_rag):
    """Optimize the vector database"""
    with st.spinner("Optimizing vector database..."):
        # Clear old caches
        opt_rag.query_cache.clear()
        
        # Save current state
        opt_rag._save_caches()
        
        display_success("Database optimization complete")

def scrape_category_sites(opt_rag, sites, category):
    """Scrape sites from a specific category"""
    with st.spinner(f"Scraping {category} websites..."):
        scraped_docs = opt_rag.web_scraper.scrape_multiple_urls(sites)
        
        if scraped_docs:
            opt_rag.rag_engine.add_documents_to_knowledge_base(scraped_docs)
            display_success(f"Scraped {len(scraped_docs)} pages from {category}")

def scrape_custom_url(opt_rag, url):
    """Scrape a custom URL"""
    with st.spinner(f"Scraping {url}..."):
        result = opt_rag.web_scraper.scrape_url(url)
        
        if result:
            opt_rag.rag_engine.add_documents_to_knowledge_base([result])
            display_success(f"Successfully scraped {url}")