
"""
RF-Expert-AI - Main Application
Universal platform for building and using domain-specific expert systems
"""
import streamlit as st
import logging
from pathlib import Path

# Import system modules
from domain_catalog import show_domain_catalog_interface, get_selected_domain
from mode_selector import show_mode_selection, get_selected_mode
from generic_rag_system import show_generic_rag_interface
from generic_finetune_system import show_generic_finetune_interface
from utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RF-Expert-AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function"""
    
    # Check if a domain is selected
    selected_domain = get_selected_domain()
    selected_mode = get_selected_mode()
    
    if selected_domain is None:
        # Show domain catalog
        show_domain_catalog_interface()
    elif selected_mode is None:
        # Show mode selection for the selected domain
        show_mode_selection()
    elif selected_mode == "rag":
        # Show generic RAG system
        show_generic_rag_interface()
    elif selected_mode == "finetune":
        # Show generic fine-tuning system
        show_generic_finetune_interface()
    else:
        st.error("Unknown mode selected")
        from domain_catalog import reset_domain_selection
        reset_domain_selection()
        st.rerun()

if __name__ == "__main__":
    main()
