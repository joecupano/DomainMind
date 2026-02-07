"""
Mode Selection Interface for RF-Expert-AI
Allows users to choose between RAG and Fine-tuning approaches for any domain
"""
import streamlit as st
from pathlib import Path
from domain_catalog import get_selected_domain, DomainCatalog

def show_mode_selection():
    """Display mode selection interface"""
    selected_domain = get_selected_domain()
    
    if not selected_domain:
        st.error("No domain selected")
        return
    
    # Get domain information
    catalog = DomainCatalog()
    domain_info = catalog.get_domain(selected_domain)
    
    if not domain_info:
        st.error(f"Domain '{selected_domain}' not found")
        return
    
    st.title(f"🔧 {domain_info['name']} Expert System")
    st.markdown(f"Choose your preferred system architecture for **{domain_info['name']}**:")
    st.markdown(f"*{domain_info['description']}*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 RAG System (Retrieval-Augmented Generation)
        
        **Best for:**
        - Immediate deployment and use
        - Large document collections
        - Frequent content updates
        - Lower computational requirements
        
        **How it works:**
        - Documents stored in vector database
        - Real-time search during queries
        - Uses base Llama model + retrieved context
        - No model training required
        
        **Advantages:**
        - Fast setup and deployment
        - Easy to add new documents
        - Source attribution for answers
        - Works with existing Llama models
        
        **Requirements:**
        - ChromaDB vector database
        - Ollama with Llama model
        - Moderate system resources
        """)
        
        if st.button("🚀 Launch RAG System", type="primary", use_container_width=True):
            st.session_state.selected_mode = "rag"
            st.rerun()
    
    with col2:
        st.markdown("""
        ### 🎯 Fine-Tuning System
        
        **Best for:**
        - Specialized RF domain expertise
        - Consistent response style
        - Offline-only operation
        - Custom model behavior
        
        **How it works:**
        - Creates specialized training dataset
        - Fine-tunes Llama on RF content
        - Direct queries to custom model
        - No retrieval step needed
        
        **Advantages:**
        - Deeply specialized knowledge
        - Faster query responses
        - No vector database needed
        - Model "learns" RF expertise
        
        **Requirements:**
        - GPU-capable system (recommended)
        - Significant training time
        - Higher computational resources
        """)
        
        if st.button("⚡ Launch Fine-Tuning System", type="secondary", use_container_width=True):
            st.session_state.selected_mode = "finetune"
            st.rerun()
    
    st.divider()
    
    # Comparison table
    st.markdown("### 📊 Detailed Comparison")
    
    comparison_data = {
        "Aspect": [
            "Setup Time",
            "Training Required",
            "Computational Needs",
            "Response Time",
            "Document Updates",
            "Accuracy",
            "Source Attribution",
            "Offline Capability"
        ],
        "RAG System": [
            "Minutes",
            "None",
            "Moderate",
            "2-5 seconds",
            "Real-time",
            "High with good docs",
            "Full source tracking",
            "After initial setup"
        ],
        "Fine-Tuning": [
            "Hours to days",
            "Required",
            "High (GPU recommended)",
            "1-2 seconds",
            "Requires retraining",
            "Very high for RF domain",
            "Limited",
            "Complete"
        ]
    }
    
    st.table(comparison_data)
    
    selected_domain = get_selected_domain()
    domain_name = domain_info['name'] if domain_info else "this domain"
    
    st.info(f"💡 **Recommendation:** Start with RAG system for immediate {domain_name.lower()} use, then consider fine-tuning for specialized deployment.")

def get_selected_mode():
    """Get the selected mode from session state"""
    return st.session_state.get('selected_mode', None)

def reset_mode_selection():
    """Reset mode selection"""
    if 'selected_mode' in st.session_state:
        del st.session_state.selected_mode