# ExpertSystem User Guide

Welcome to ExpertSystem, a universal platform for building and using domain-specific expert systems. This guide will help you create, configure, and use expert systems for any specialized domain.

## What is ExpertSystem?

ExpertSystem allows you to build AI-powered expert systems for any specialized field:
- Medical diagnostics
- Legal research
- RF communications
- Amateur radio
- Scientific research
- Engineering
- Financial analysis
- Education
- Any specialized field

## Getting Started

### Accessing the Application

Once installed, access ExpertSystem through your web browser:

- **Standard Installation**: `http://your-server-ip` (port 80/443)
- **Docker Deployment**: `http://your-server-ip:8080`
- **Development Setup**: `http://localhost:8501`

### First Steps

1. **Create Your First Domain** - Set up a specialized expert system
2. **Choose Architecture** - Select RAG or Fine-tuning approach
3. **Add Knowledge** - Upload documents or scrape websites
4. **Start Querying** - Ask questions and get expert answers

---

## Domain Catalog

The Domain Catalog is your hub for managing multiple expert systems.

### Creating a New Domain

1. Click **"Create Domain"** in the sidebar
2. Fill in domain details:
   - **Name**: Descriptive name (e.g., "Medical Diagnostics", "Amateur Radio Technical Reference")
   - **Keywords**: Relevant terms for ranking (e.g., "diagnosis, symptoms, treatment")
   - **Websites**: Authoritative sources for web scraping (one per line)
3. Click **"Create Domain"** to save

**Example - Amateur Radio Domain:**
```
Name: Amateur Radio Technical Reference
Keywords: ham radio, antenna design, propagation, modulation, transceiver, repeater, licensing, QSO
Websites:
https://www.arrl.org
https://www.qrz.com
http://www.hamuniverse.com
```

**Example - Medical Diagnostics Domain:**
```
Name: Medical Diagnostics
Keywords: diagnosis, symptoms, treatment, medicine, pathology, clinical
Websites:
https://www.ncbi.nlm.nih.gov
https://www.mayoclinic.org
https://medlineplus.gov
```

### Managing Domains

- **Browse Domains**: View all created expert systems in the catalog
- **Select Domain**: Click on a domain to activate it
- **Switch Domains**: Change between different expert systems at any time
- **View Details**: See domain configuration, keywords, and statistics

---

## Choosing an Architecture

ExpertSystem offers two approaches for building expert systems:

### RAG System (Recommended for Most Users)

**Best for:**
- ✅ Immediate deployment
- ✅ Frequently updated content
- ✅ Large document collections
- ✅ Source attribution needs
- ✅ Quick setup (minutes)

**How it works:**
Documents are stored in a vector database and retrieved when needed to answer questions.

**Setup time:** 5-10 minutes

### Fine-Tuning System

**Best for:**
- Specialized domain expertise
- Consistent response style
- Offline-only operation
- Stable document collections
- Custom model behavior

**How it works:**
Creates a custom AI model trained specifically on your domain knowledge.

**Setup time:** Several hours to days (includes training)

### Comparison Table

| Feature | RAG System | Fine-Tuning System |
|---------|------------|-------------------|
| **Setup Time** | Minutes | Hours to Days |
| **Updates** | Real-time | Requires retraining |
| **Response Time** | 2-5 seconds | 1-2 seconds |
| **Hardware Needs** | Moderate | High (GPU recommended) |
| **Source Tracking** | Full | Limited |
| **Offline Use** | After setup | Complete |
| **Expertise Level** | Good | Excellent |

---

## Using the RAG System

### Adding Knowledge to Your Domain

#### 1. Upload Documents

Navigate to **"Document Management"** tab:

1. Click **"Browse files"** to select documents
2. Supported formats: PDF, DOCX, DOC, TXT
3. Upload multiple files at once for batch processing
4. Wait for processing to complete
5. View document statistics in sidebar

**Tips:**
- Organize documents by topic for better results
- Use clear, well-formatted documents
- Include technical manuals, textbooks, research papers
- Avoid scanned images (OCR not supported yet)

#### 2. Web Scraping

Navigate to **"Web Scraping"** tab:

**Using Domain Websites:**
1. Review pre-configured websites for your domain
2. Click **"Scrape Domain Websites"**
3. Wait for content extraction
4. View scraped content statistics

**Custom URLs:**
1. Enter specific URLs (one per line)
2. Click **"Scrape Custom URLs"**
3. System extracts and processes content automatically

**Tips:**
- Use authoritative sources for your domain
- Respect website terms of service
- Allow sufficient time for large websites
- Check scraped content quality in logs

### Asking Questions

Navigate to **"Ask Questions"** tab:

1. Type your question in the text box
2. Press Enter or click **"Submit"**
3. View AI-generated answer with sources
4. Review source documents and relevance scores

**Quick Topic Buttons:**
Domain-specific buttons for common queries (customizable per domain)

**Conversation History:**
- Previous exchanges are remembered for context
- Clear history with **"Clear Conversation"** button
- Export conversations for reference

**Example Questions:**

*Amateur Radio Domain:*
- "What is the difference between FM and SSB modulation?"
- "How do I calculate antenna length for 2 meters?"
- "What are the licensing requirements for HF operation?"

*Medical Diagnostics Domain:*
- "What are the symptoms of type 2 diabetes?"
- "How is pneumonia diagnosed?"
- "What are contraindications for aspirin?"

### Advanced Features

#### Performance Analytics

View real-time statistics in the sidebar:
- **Cache Hit Rate**: Percentage of cached responses (higher is better)
- **Average Query Time**: Response speed metrics
- **Document Count**: Total knowledge base size
- **System Health**: Overall performance indicator

#### Export Functionality

Save conversations and answers:
- **Export Conversation**: Download full chat history
- **Copy to Clipboard**: Quick copy for sharing
- **Source References**: Include document citations

---

## Using the Fine-Tuning System

### Creating Training Data

Navigate to **"Data Preparation"** tab:

#### 1. Upload Domain Documents

1. Upload documents containing domain knowledge
2. System analyzes content and extracts key information
3. View document processing statistics

#### 2. Generate Training Dataset

1. Click **"Generate Training Dataset"**
2. System creates question-answer pairs automatically
3. Domain-specific patterns ensure relevant questions
4. Review generated dataset quality

**Dataset Contents:**
- Instruction-response pairs
- Domain-specific questions
- Context-aware answers
- Training-ready format

#### 3. Download Training Data

1. Click **"Download Training Dataset"**
2. Save JSON file for training
3. Use with provided training scripts

### Training Configuration

Navigate to **"Training Configuration"** tab:

#### Configure Parameters

**Basic Settings:**
- **Epochs**: Training iterations (recommended: 3-5)
- **Learning Rate**: Training speed (default: 2e-4)
- **Batch Size**: Memory vs speed trade-off (default: 4)
- **Base Model**: Choose Llama model size (7B or 13B)

**Advanced Settings:**
- **LoRA Rank**: Model adaptation complexity (default: 16)
- **LoRA Alpha**: Scaling parameter (default: 32)
- **Max Sequence Length**: Input size limit (default: 2048)

#### Generate Training Scripts

1. Configure parameters above
2. Click **"Generate Training Scripts"**
3. Download complete training package:
   - Python training script
   - Requirements file
   - Configuration file
   - Instructions

### Running Training

**Requirements:**
- GPU with 16GB+ VRAM (recommended)
- CUDA support
- PyTorch with GPU support
- Several hours for training

**Steps:**
1. Extract downloaded training package
2. Install requirements: `pip install -r requirements.txt`
3. Run training script: `python train_model.py`
4. Monitor training progress
5. Wait for completion

**Note:** Training is computationally intensive and best done on dedicated hardware.

### Deploying Fine-Tuned Models

After training completes:

1. **Convert Model** for Ollama:
   ```bash
   ollama create my-domain-expert -f Modelfile
   ```

2. **Test Model**:
   ```bash
   ollama run my-domain-expert
   ```

3. **Use in ExpertSystem**:
   - Update model name in configuration
   - Restart application
   - Start querying with your custom model

---

## Best Practices

### For Best Results

1. **Quality Documents**:
   - Use authoritative sources
   - Include diverse perspectives
   - Keep content relevant to domain
   - Update regularly

2. **Clear Questions**:
   - Be specific and detailed
   - Use domain terminology
   - Provide context when needed
   - Build on conversation history

3. **Domain Configuration**:
   - Choose relevant keywords
   - Select authoritative websites
   - Organize by sub-topics if needed
   - Review and refine over time

### Performance Optimization

1. **RAG System**:
   - Upload documents in batches
   - Use web scraping for bulk content
   - Monitor cache hit rates
   - Clear old conversations periodically

2. **Fine-Tuning System**:
   - Provide diverse training examples
   - Balance dataset across topics
   - Test model before full deployment
   - Retrain when domain knowledge updates

---

## 🔧 Tips and Tricks

### Effective Document Selection

- **Technical Manuals**: Official documentation and specifications
- **Academic Papers**: Peer-reviewed research
- **Textbooks**: Comprehensive foundational knowledge
- **Standards**: Industry standards and regulations
- **Tutorials**: Practical how-to guides

### Writing Better Queries

**Instead of:** "Tell me about antennas"
**Try:** "What are the key design considerations for a 2-meter Yagi antenna?"

**Instead of:** "Heart disease information"
**Try:** "What are the diagnostic criteria for coronary artery disease?"

### Managing Multiple Domains

1. **Naming**: Use clear, descriptive names
2. **Keywords**: Keep keyword lists focused and relevant
3. **Organization**: Group related domains
4. **Maintenance**: Regularly review and update
5. **Switching**: Use domain selector in sidebar

---

## 📊 Understanding Results

### RAG System Responses

Each response includes:

- **Answer**: AI-generated response based on your documents
- **Sources**: Documents used to generate answer
- **Relevance Scores**: How well sources match your question
- **Conversation Context**: Reference to previous exchanges

**Interpreting Scores:**
- **> 0.8**: Highly relevant source
- **0.6 - 0.8**: Moderately relevant
- **< 0.6**: Tangentially related

### Fine-Tuned Model Responses

Responses from fine-tuned models:

- **Direct answers**: No source attribution
- **Consistent style**: Learned from training data
- **Domain expertise**: Deep understanding of terminology
- **Faster responses**: No retrieval needed

---

## 🆘 Common Questions

### "My question returns no results"

- Check if relevant documents are uploaded
- Try broader search terms
- Verify domain is selected correctly
- Add more documents to knowledge base

### "Responses are generic"

- Upload more domain-specific documents
- Add technical resources via web scraping
- Refine domain keywords
- Consider fine-tuning for deeper expertise

### "Processing is slow"

- This is normal for large documents
- Use batch upload for multiple files
- Check system resources in admin panel
- Consider hardware upgrades for fine-tuning

### "How do I switch between RAG and Fine-tuning?"

- Return to mode selection in sidebar
- Choose different architecture
- Existing data is preserved
- Can switch back anytime

---

## 📞 Getting Help

### In-Application Help

- **System Health**: Check status in sidebar
- **Performance Metrics**: View analytics dashboard
- **Error Messages**: Read detailed error descriptions
- **Logs**: Review processing logs in interface

### Documentation

- **README.md**: Overview and features
- **INSTALLATION.md**: Setup and configuration
- **ADMINISTRATION.md**: System management
- **DEVELOPER.md**: Technical details

### Support Resources

- Check system logs for detailed errors
- Review documentation for specific features
- Verify system requirements are met
- Ensure services are running properly

---

## 🎓 Example Workflows

### Workflow 1: Building a Medical Expert System

1. **Create Domain**: "Medical Diagnostics"
2. **Add Keywords**: diagnosis, symptoms, treatment, pathology
3. **Upload Documents**: Medical textbooks, clinical guidelines
4. **Scrape Websites**: PubMed, Mayo Clinic, CDC
5. **Start Querying**: Ask about symptoms, treatments, diagnoses
6. **Refine**: Add more specialized documents as needed

### Workflow 2: Amateur Radio Technical Reference

1. **Create Domain**: "Amateur Radio Technical Reference"
2. **Add Keywords**: ham radio, antenna, propagation, modulation
3. **Upload Documents**: ARRL handbooks, FCC regulations
4. **Scrape Websites**: ARRL.org, QRZ.com
5. **Start Querying**: Technical questions about equipment, licensing
6. **Expand**: Add specialized topics (antennas, digital modes, etc.)

### Workflow 3: Legal Research Assistant

1. **Create Domain**: "Legal Research"
2. **Add Keywords**: law, case, statute, regulation, precedent
3. **Upload Documents**: Case law, regulations, legal textbooks
4. **Fine-Tune**: Create specialized legal reasoning model
5. **Deploy**: Use for legal analysis and research
6. **Update**: Retrain as new precedents emerge

---

**Ready to build your expert system? Start by creating your first domain!**
