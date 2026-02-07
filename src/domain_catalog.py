"""
Domain Catalog Manager for RF-Expert-AI
Manages multiple domain-specific expert systems and their configurations
"""
import streamlit as st
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from utils import display_error, display_success, display_info, display_warning

class DomainCatalog:
    """Manages catalog of domain-specific expert systems"""
    
    def __init__(self):
        self.catalog_dir = Path("data/domains")
        self.catalog_dir.mkdir(exist_ok=True)
        self.catalog_file = self.catalog_dir / "catalog.json"
        self.domains = self._load_catalog()
    
    def _load_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load domain catalog from file"""
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                display_warning(f"Could not load catalog: {e}")
        
        # Return default RF domain for backward compatibility
        return {
            "rf_communications": {
                "name": "RF Communications",
                "description": "Expert system for radio frequency communications, antenna design, circuit analysis, and RF measurements",
                "keywords": ["antenna", "frequency", "impedance", "amplifier", "filter", "transmission", "power", "noise", "circuit", "signal"],
                "websites": [
                    "https://www.rfcafe.com",
                    "https://www.microwaves101.com", 
                    "https://www.antenna-theory.com"
                ],
                "created_date": "2025-07-20",
                "status": "active",
                "documents_count": 0,
                "model_type": "base"
            }
        }
    
    def _save_catalog(self):
        """Save catalog to file"""
        try:
            with open(self.catalog_file, 'w', encoding='utf-8') as f:
                json.dump(self.domains, f, indent=2, ensure_ascii=False)
        except Exception as e:
            display_error(f"Could not save catalog: {e}")
    
    def add_domain(self, domain_id: str, name: str, description: str, 
                   keywords: List[str], websites: List[str]) -> bool:
        """Add new domain to catalog"""
        if domain_id in self.domains:
            display_warning(f"Domain '{domain_id}' already exists")
            return False
        
        self.domains[domain_id] = {
            "name": name,
            "description": description,
            "keywords": keywords,
            "websites": websites,
            "created_date": time.strftime("%Y-%m-%d"),
            "status": "active",
            "documents_count": 0,
            "model_type": "base"
        }
        
        # Create domain-specific directories
        domain_dir = self.catalog_dir / domain_id
        domain_dir.mkdir(exist_ok=True)
        (domain_dir / "documents").mkdir(exist_ok=True)
        (domain_dir / "scraped").mkdir(exist_ok=True)
        (domain_dir / "vector_db").mkdir(exist_ok=True)
        (domain_dir / "cache").mkdir(exist_ok=True)
        (domain_dir / "training_data").mkdir(exist_ok=True)
        (domain_dir / "models").mkdir(exist_ok=True)
        
        self._save_catalog()
        display_success(f"Added domain: {name}")
        return True
    
    def update_domain(self, domain_id: str, **kwargs) -> bool:
        """Update domain configuration"""
        if domain_id not in self.domains:
            display_error(f"Domain '{domain_id}' not found")
            return False
        
        for key, value in kwargs.items():
            if key in self.domains[domain_id]:
                self.domains[domain_id][key] = value
        
        self._save_catalog()
        return True
    
    def delete_domain(self, domain_id: str) -> bool:
        """Delete domain from catalog"""
        if domain_id not in self.domains:
            display_error(f"Domain '{domain_id}' not found")
            return False
        
        # Mark as deleted rather than removing
        self.domains[domain_id]["status"] = "deleted"
        self._save_catalog()
        display_success(f"Domain '{domain_id}' marked as deleted")
        return True
    
    def get_active_domains(self) -> Dict[str, Dict[str, Any]]:
        """Get all active domains"""
        return {k: v for k, v in self.domains.items() if v.get("status") == "active"}
    
    def get_domain(self, domain_id: str) -> Optional[Dict[str, Any]]:
        """Get specific domain configuration"""
        return self.domains.get(domain_id)
    
    def get_domain_path(self, domain_id: str) -> Path:
        """Get path to domain directory"""
        return self.catalog_dir / domain_id
    
    def get_domain_stats(self, domain_id: str) -> Dict[str, Any]:
        """Get domain statistics"""
        if domain_id not in self.domains:
            return {}
        
        domain_path = self.get_domain_path(domain_id)
        
        # Count files in each directory
        stats = {
            "documents": len(list((domain_path / "documents").glob("*"))),
            "scraped_files": len(list((domain_path / "scraped").glob("*"))),
            "training_datasets": len(list((domain_path / "training_data").glob("*.json"))),
            "trained_models": len(list((domain_path / "models").glob("*"))),
            "status": self.domains[domain_id].get("status", "unknown"),
            "created_date": self.domains[domain_id].get("created_date", "unknown")
        }
        
        return stats

def show_domain_catalog_interface():
    """Main interface for domain catalog management"""
    st.header("🏛️ Domain Expert System Catalog")
    st.markdown("Manage and select domain-specific expert systems")
    
    # Initialize catalog
    if 'domain_catalog' not in st.session_state:
        st.session_state.domain_catalog = DomainCatalog()
    
    catalog = st.session_state.domain_catalog
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Browse Domains", "➕ Create Domain", "⚙️ Manage Domains"])
    
    with tab1:
        show_domain_browser(catalog)
    
    with tab2:
        show_domain_creator(catalog)
    
    with tab3:
        show_domain_manager(catalog)

def show_domain_browser(catalog):
    """Browse and select domains"""
    st.subheader("Available Expert Systems")
    
    active_domains = catalog.get_active_domains()
    
    if not active_domains:
        st.info("No active domains available. Create a new domain to get started.")
        return
    
    # Domain selection grid
    cols = st.columns(2)
    
    for i, (domain_id, domain_info) in enumerate(active_domains.items()):
        with cols[i % 2]:
            with st.container():
                st.write(f"### {domain_info['name']}")
                st.write(domain_info['description'])
                
                # Domain statistics
                stats = catalog.get_domain_stats(domain_id)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", stats.get('documents', 0))
                with col2:
                    st.metric("Web Sources", stats.get('scraped_files', 0))
                with col3:
                    st.metric("Models", stats.get('trained_models', 0))
                
                # Keywords
                if domain_info.get('keywords'):
                    st.write("**Keywords:**", ", ".join(domain_info['keywords'][:5]))
                
                # Launch buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔍 Launch RAG", key=f"rag_{domain_id}"):
                        st.session_state.selected_domain = domain_id
                        st.session_state.selected_mode = "rag"
                        st.rerun()
                
                with col2:
                    if st.button(f"🎯 Launch Fine-tuning", key=f"ft_{domain_id}"):
                        st.session_state.selected_domain = domain_id
                        st.session_state.selected_mode = "finetune"
                        st.rerun()
                
                st.divider()

def show_domain_creator(catalog):
    """Create new domain interface"""
    st.subheader("Create New Expert System Domain")
    
    with st.form("create_domain"):
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            domain_name = st.text_input("Domain Name*", placeholder="e.g., Medical Diagnostics")
            domain_id = st.text_input("Domain ID*", placeholder="e.g., medical_diagnostics", help="Unique identifier (lowercase, underscores only)")
        
        with col2:
            domain_description = st.text_area("Description*", placeholder="Detailed description of the expert system domain and its capabilities")
        
        # Keywords
        st.subheader("Domain Keywords")
        keywords_text = st.text_area(
            "Keywords (one per line)*",
            placeholder="diagnosis\nsymptoms\ntreatment\nmedicine\ndisease",
            help="Key terms that define this domain - used for content relevance scoring"
        )
        
        # Default websites
        st.subheader("Default Information Sources")
        websites_text = st.text_area(
            "Websites (one per line)",
            placeholder="https://www.ncbi.nlm.nih.gov\nhttps://www.mayoclinic.org\nhttps://www.webmd.com",
            help="Authoritative websites for this domain (optional)"
        )
        
        # Submit button
        submitted = st.form_submit_button("Create Expert System Domain", type="primary")
        
        if submitted:
            # Validation
            if not domain_name or not domain_id or not domain_description:
                st.error("Please fill in all required fields")
                return
            
            if not domain_id.replace("_", "").isalnum():
                st.error("Domain ID must contain only letters, numbers, and underscores")
                return
            
            # Process keywords
            keywords = [kw.strip() for kw in keywords_text.split('\n') if kw.strip()]
            if not keywords:
                st.error("Please provide at least one keyword")
                return
            
            # Process websites
            websites = [url.strip() for url in websites_text.split('\n') if url.strip()]
            
            # Create domain
            if catalog.add_domain(domain_id, domain_name, domain_description, keywords, websites):
                st.success(f"Successfully created expert system: {domain_name}")
                time.sleep(1)
                st.rerun()

def show_domain_manager(catalog):
    """Manage existing domains"""
    st.subheader("Domain Management")
    
    active_domains = catalog.get_active_domains()
    
    if not active_domains:
        st.info("No domains to manage")
        return
    
    # Domain selection
    selected_domain_id = st.selectbox(
        "Select Domain to Manage",
        options=list(active_domains.keys()),
        format_func=lambda x: active_domains[x]['name']
    )
    
    if selected_domain_id:
        domain_info = catalog.get_domain(selected_domain_id)
        stats = catalog.get_domain_stats(selected_domain_id)
        
        # Domain overview
        st.subheader(f"Managing: {domain_info['name']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Description:**", domain_info['description'])
            st.write("**Created:**", domain_info['created_date'])
            st.write("**Status:**", domain_info['status'])
        
        with col2:
            st.write("**Statistics:**")
            st.write(f"- Documents: {stats.get('documents', 0)}")
            st.write(f"- Web sources: {stats.get('scraped_files', 0)}")
            st.write(f"- Training datasets: {stats.get('training_datasets', 0)}")
            st.write(f"- Trained models: {stats.get('trained_models', 0)}")
        
        # Keywords
        st.write("**Keywords:**", ", ".join(domain_info['keywords']))
        
        # Websites
        if domain_info.get('websites'):
            st.write("**Default Websites:**")
            for website in domain_info['websites']:
                st.write(f"- {website}")
        
        # Management actions
        st.subheader("Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Edit Domain", type="secondary"):
                st.info("Edit functionality coming soon")
        
        with col2:
            if st.button("📊 View Analytics", type="secondary"):
                st.info("Analytics dashboard coming soon")
        
        with col3:
            if st.button("🗑️ Delete Domain", type="secondary"):
                if st.checkbox(f"Confirm deletion of {domain_info['name']}"):
                    catalog.delete_domain(selected_domain_id)
                    st.rerun()

def get_selected_domain():
    """Get currently selected domain"""
    return st.session_state.get('selected_domain', None)

def reset_domain_selection():
    """Reset domain selection"""
    if 'selected_domain' in st.session_state:
        del st.session_state.selected_domain
    if 'selected_mode' in st.session_state:
        del st.session_state.selected_mode