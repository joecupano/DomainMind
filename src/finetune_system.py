"""
Fine-Tuning System for RF Communications Expert System
Creates training datasets and fine-tunes Llama models on RF content
"""
import streamlit as st
import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import time

from document_processor import DocumentProcessor
from web_scraper import WebScraper
from utils import (
    display_error, display_success, display_info, display_warning,
    format_file_size, clean_text
)
from config import DOCUMENTS_DIR, SCRAPED_DIR, DATA_DIR

logger = logging.getLogger(__name__)

class FineTuningSystem:
    """Handles fine-tuning workflow for RF communications"""
    
    def __init__(self):
        self.training_data_dir = DATA_DIR / "training_data"
        self.training_data_dir.mkdir(exist_ok=True)
        self.model_dir = DATA_DIR / "finetuned_models"
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize processors
        self.document_processor = DocumentProcessor()
        self.web_scraper = WebScraper()
        
        # Training parameters
        self.training_config = {
            "base_model": "llama2-7b",
            "epochs": 3,
            "learning_rate": 2e-4,
            "batch_size": 4,
            "max_seq_length": 2048,
            "lora_rank": 16,
            "lora_alpha": 32
        }
    
    def create_training_dataset(self, documents: List[Dict[str, Any]]) -> str:
        """Convert documents to training format"""
        training_examples = []
        
        for doc in documents:
            # Create instruction-response pairs from RF content
            chunks = doc.get('chunks', [])
            source_type = doc.get('source_type', 'document')
            
            for i, chunk in enumerate(chunks):
                # Generate RF-specific questions for each chunk
                questions = self._generate_questions_for_chunk(chunk, source_type)
                
                for question in questions:
                    training_example = {
                        "instruction": f"Answer this RF communications question: {question}",
                        "input": "",
                        "output": chunk
                    }
                    training_examples.append(training_example)
        
        # Save training dataset
        timestamp = int(time.time())
        dataset_file = self.training_data_dir / f"rf_training_dataset_{timestamp}.json"
        
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(training_examples, f, indent=2, ensure_ascii=False)
        
        display_success(f"Created training dataset with {len(training_examples)} examples")
        return str(dataset_file)
    
    def _generate_questions_for_chunk(self, chunk: str, source_type: str) -> List[str]:
        """Generate relevant questions for RF content chunk"""
        questions = []
        
        # RF-specific keywords and their associated questions
        rf_keywords = {
            "antenna": [
                "What are the key antenna parameters?",
                "How do you design an antenna?",
                "What factors affect antenna performance?"
            ],
            "frequency": [
                "How does frequency affect RF design?",
                "What frequency considerations are important?",
                "How do you select operating frequency?"
            ],
            "impedance": [
                "What is impedance matching?",
                "How do you achieve proper impedance matching?",
                "Why is impedance important in RF circuits?"
            ],
            "amplifier": [
                "How do RF amplifiers work?",
                "What are the key amplifier specifications?",
                "How do you design an RF amplifier?"
            ],
            "filter": [
                "What types of RF filters exist?",
                "How do you design RF filters?",
                "What filter specifications matter?"
            ],
            "transmission": [
                "How do transmission lines work?",
                "What transmission line parameters matter?",
                "How do you calculate transmission line properties?"
            ],
            "power": [
                "How do you measure RF power?",
                "What power considerations are important?",
                "How do you handle RF power safely?"
            ],
            "noise": [
                "What causes RF noise?",
                "How do you measure noise figure?",
                "How do you minimize RF noise?"
            ]
        }
        
        # Find relevant keywords in chunk
        chunk_lower = chunk.lower()
        for keyword, keyword_questions in rf_keywords.items():
            if keyword in chunk_lower:
                questions.extend(keyword_questions[:2])  # Limit to 2 questions per keyword
        
        # Add generic RF questions
        if not questions:
            questions = [
                "Explain this RF communications concept.",
                "What are the key points about this RF topic?",
                "How does this apply to RF engineering?"
            ]
        
        return questions[:3]  # Limit to 3 questions per chunk
    
    def generate_training_script(self, dataset_path: str) -> str:
        """Generate training script for fine-tuning"""
        config_json = json.dumps(self.training_config, indent=4)
        
        script_content = f'''#!/usr/bin/env python3
"""
RF Communications Fine-Tuning Script
Generated automatically from RF expert system
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
from peft import LoraConfig, get_peft_model, TaskType
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training configuration
CONFIG = {config_json}

def load_dataset(dataset_path: str):
    """Load and prepare training dataset"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Format data for instruction tuning
    formatted_data = []
    for item in data:
        text = f"### Instruction: {{item['instruction']}}\\n### Response: {{item['output']}}"
        formatted_data.append({{"text": text}})
    
    return Dataset.from_list(formatted_data)

def setup_model_and_tokenizer():
    """Initialize model and tokenizer"""
    model_name = "meta-llama/Llama-2-7b-hf"  # Adjust based on availability
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return model, tokenizer

def setup_lora(model):
    """Configure LoRA for efficient fine-tuning"""
    lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    return model

def main():
    """Main training function"""
    logger.info("Starting RF Communications Fine-Tuning")
    
    # Load dataset
    dataset = load_dataset("{dataset_path}")
    logger.info(f"Loaded {{len(dataset)}} training examples")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    model = setup_lora(model)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=CONFIG["max_seq_length"]
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./rf_finetuned_model",
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=2,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="no",
        learning_rate=CONFIG["learning_rate"],
        fp16=True,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained("./rf_finetuned_model")
    
    logger.info("Fine-tuning completed!")

if __name__ == "__main__":
    main()
'''
        
        script_file = self.training_data_dir / "finetune_rf_model.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Also create requirements file
        requirements = '''torch>=1.9.0
transformers>=4.21.0
datasets>=2.0.0
peft>=0.3.0
accelerate>=0.20.0
bitsandbytes>=0.39.0
scipy>=1.7.0
'''
        
        req_file = self.training_data_dir / "requirements_finetune.txt"
        with open(req_file, 'w') as f:
            f.write(requirements)
        
        display_success(f"Generated training script: {script_file}")
        return str(script_file)
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and files"""
        dataset_files = list(self.training_data_dir.glob("*.json"))
        script_files = list(self.training_data_dir.glob("*.py"))
        model_dirs = [d for d in self.model_dir.iterdir() if d.is_dir()]
        
        return {
            "datasets_created": len(dataset_files),
            "training_scripts": len(script_files),
            "trained_models": len(model_dirs),
            "latest_dataset": dataset_files[-1] if dataset_files else None,
            "latest_script": script_files[-1] if script_files else None
        }

def show_finetune_interface():
    """Main interface for fine-tuning system"""
    st.header("🎯 Fine-Tuning System for RF Communications")
    st.markdown("Create specialized RF models through fine-tuning")
    
    # Initialize fine-tuning system
    if 'finetune_system' not in st.session_state:
        st.session_state.finetune_system = FineTuningSystem()
    
    finetune_sys = st.session_state.finetune_system
    
    # Sidebar for status and configuration
    with st.sidebar:
        st.header("🔧 Fine-Tuning Status")
        status = finetune_sys.get_training_status()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Datasets", status['datasets_created'])
            st.metric("Scripts", status['training_scripts'])
        with col2:
            st.metric("Models", status['trained_models'])
        
        st.divider()
        
        st.subheader("⚙️ Training Config")
        finetune_sys.training_config["epochs"] = st.slider("Epochs", 1, 10, 3)
        finetune_sys.training_config["learning_rate"] = st.select_slider(
            "Learning Rate", 
            options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4],
            value=2e-4,
            format_func=lambda x: f"{x:.0e}"
        )
        finetune_sys.training_config["batch_size"] = st.selectbox("Batch Size", [1, 2, 4, 8], index=2)
        
        if st.button("🔄 Reset Mode", type="secondary"):
            from mode_selector import reset_mode_selection
            reset_mode_selection()
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Data Preparation", "🎯 Training", "🧪 Testing", "📋 Instructions"])
    
    with tab1:
        show_data_preparation(finetune_sys)
    
    with tab2:
        show_training_interface(finetune_sys)
    
    with tab3:
        show_testing_interface(finetune_sys)
    
    with tab4:
        show_training_instructions()

def show_data_preparation(finetune_sys):
    """Data preparation interface"""
    st.subheader("📄 Document Processing")
    
    # Document upload
    uploaded_files = st.file_uploader(
        "Upload RF documents for training",
        type=['pdf', 'docx', 'doc', 'txt'],
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_files and st.button("📥 Process Documents", type="primary"):
            process_documents_for_training(finetune_sys, uploaded_files)
    
    with col2:
        if st.button("🌐 Scrape RF Websites", type="secondary"):
            scrape_content_for_training(finetune_sys)
    
    # Show existing documents
    st.subheader("📁 Existing Training Data")
    doc_files = list(DOCUMENTS_DIR.glob("*"))
    scraped_files = list(SCRAPED_DIR.glob("*"))
    
    if doc_files or scraped_files:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Documents:**")
            for file in doc_files[:5]:
                st.write(f"• {file.name}")
        
        with col2:
            st.write("**Scraped Content:**")
            for file in scraped_files[:5]:
                st.write(f"• {file.name}")
        
        if st.button("🔄 Create Training Dataset", type="primary"):
            create_full_training_dataset(finetune_sys)
    else:
        st.info("Upload documents or scrape websites to create training data")

def show_training_interface(finetune_sys):
    """Training interface"""
    st.subheader("🚀 Model Training")
    
    status = finetune_sys.get_training_status()
    
    if status['datasets_created'] == 0:
        st.warning("Create a training dataset first in the Data Preparation tab")
        return
    
    st.success(f"Training dataset ready with {status['datasets_created']} versions")
    
    if status['latest_script']:
        st.info(f"Training script available: {status['latest_script'].name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Training Script", type="primary"):
                with open(status['latest_script'], 'r') as f:
                    st.download_button(
                        "Download Script",
                        f.read(),
                        file_name="finetune_rf_model.py",
                        mime="text/python"
                    )
        
        with col2:
            if st.button("📋 View Training Commands"):
                show_training_commands()

def show_testing_interface(finetune_sys):
    """Model testing interface"""
    st.subheader("🧪 Model Testing")
    
    # Model selection
    model_dirs = [d for d in finetune_sys.model_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        st.info("No trained models available yet")
        return
    
    selected_model = st.selectbox("Select Model", [d.name for d in model_dirs])
    
    # Test queries
    st.subheader("Test Queries")
    test_query = st.text_input("Enter RF question:")
    
    if test_query and st.button("Test Model"):
        st.info("Model testing requires the trained model to be loaded locally")
        st.code(f"# Test query: {test_query}")

def show_training_instructions():
    """Show detailed training instructions"""
    st.subheader("📋 Training Instructions")
    
    st.markdown("""
    ### Prerequisites
    
    1. **Hardware Requirements:**
       - GPU recommended (NVIDIA with CUDA support)
       - Minimum 16GB RAM (32GB+ recommended)
       - 50GB+ free disk space
    
    2. **Software Requirements:**
       - Python 3.8+
       - CUDA toolkit (if using GPU)
       - Git and Git LFS
    
    ### Training Steps
    
    1. **Prepare Environment:**
    ```bash
    # Create virtual environment
    python -m venv rf_finetune_env
    source rf_finetune_env/bin/activate  # Linux/Mac
    # or
    rf_finetune_env\\Scripts\\activate  # Windows
    
    # Install requirements
    pip install -r requirements_finetune.txt
    ```
    
    2. **Download Base Model:**
    ```bash
    # Using Hugging Face CLI
    huggingface-cli login
    huggingface-cli download meta-llama/Llama-2-7b-hf
    ```
    
    3. **Run Training:**
    ```bash
    python finetune_rf_model.py
    ```
    
    4. **Monitor Training:**
    - Watch for loss reduction
    - Check GPU memory usage
    - Monitor training logs
    
    ### Expected Timeline
    
    - **7B Model**: 4-8 hours on single GPU
    - **13B Model**: 8-16 hours on single GPU
    - **Multiple GPUs**: Proportionally faster
    
    ### Post-Training
    
    1. **Model Conversion:**
    ```bash
    # Convert to GGML for Ollama
    python convert_to_ggml.py rf_finetuned_model
    ```
    
    2. **Load in Ollama:**
    ```bash
    # Create Modelfile
    echo "FROM ./rf_finetuned_model.ggml" > Modelfile
    ollama create rf-expert -f Modelfile
    ```
    
    3. **Test Model:**
    ```bash
    ollama run rf-expert "What is antenna gain?"
    ```
    """)

def process_documents_for_training(finetune_sys, uploaded_files):
    """Process uploaded documents for training"""
    processed_docs = []
    
    for uploaded_file in uploaded_files:
        result = finetune_sys.document_processor.process_uploaded_file(uploaded_file)
        if result:
            processed_docs.append(result)
    
    if processed_docs:
        dataset_file = finetune_sys.create_training_dataset(processed_docs)
        script_file = finetune_sys.generate_training_script(dataset_file)
        display_success(f"Processed {len(processed_docs)} documents for training")

def scrape_content_for_training(finetune_sys):
    """Scrape content for training"""
    scraped_docs = finetune_sys.web_scraper.scrape_rf_websites()
    
    if scraped_docs:
        dataset_file = finetune_sys.create_training_dataset(scraped_docs)
        script_file = finetune_sys.generate_training_script(dataset_file)
        display_success(f"Scraped {len(scraped_docs)} documents for training")

def create_full_training_dataset(finetune_sys):
    """Create training dataset from all available content"""
    # Process existing documents
    all_docs = []
    
    # Load existing documents
    for doc_file in DOCUMENTS_DIR.glob("*"):
        result = finetune_sys.document_processor.process_file(doc_file)
        if result:
            all_docs.append(result)
    
    # Load scraped content
    for scraped_file in SCRAPED_DIR.glob("*.txt"):
        content = finetune_sys.web_scraper.load_scraped_content(scraped_file)
        if content:
            all_docs.append(content)
    
    if all_docs:
        dataset_file = finetune_sys.create_training_dataset(all_docs)
        script_file = finetune_sys.generate_training_script(dataset_file)
        display_success(f"Created comprehensive training dataset from {len(all_docs)} sources")

def show_training_commands():
    """Show training command examples"""
    st.code("""
# Basic training
python finetune_rf_model.py

# With custom parameters
python finetune_rf_model.py --epochs 5 --batch_size 2

# Multi-GPU training
torchrun --nproc_per_node=2 finetune_rf_model.py

# Monitor training
tensorboard --logdir ./runs
    """, language="bash")