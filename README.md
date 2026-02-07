# RF Expert AI

**Enterprise-Grade AI Platform for Highly Regulated Environments**

RF Expert AI is a secure, on-premise platform for building and deploying domain-specific expert systems with comprehensive audit logging and chain of custody tracking. Designed for regulated industries including healthcare, finance, government, and defense.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Ubuntu%2024.04-orange.svg)](https://ubuntu.com/)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://docker.com)
[![Security](https://img.shields.io/badge/security-SOC2%20%7C%20GDPR%20%7C%20HIPAA-red.svg)](docs/SECURITY_POLICY.md)

---

## Enterprise Security Features

### Network Isolation
- **Segmented Networks**: Frontend and backend network isolation
- **Zero External Access**: Backend services cannot reach internet
- **Minimal Attack Surface**: Only essential ports exposed

### Chain of Custody
- **Blockchain-Style Audit Trail**: Cryptographically chained events
- **Immutable Logs**: Deletion-protected audit database
- **7-Year Retention**: Compliance-ready log retention
- **Complete Tracking**: Every action, user, and data access logged

### Container Security
- **Read-Only Filesystems**: Prevents runtime modifications
- **Dropped Capabilities**: Minimal Linux capabilities
- **Non-Root Users**: All containers run as unprivileged users
- **Resource Limits**: CPU and memory quotas enforced

### Compliance Ready
- **SOC 2 Type II**: Access controls, audit logging, monitoring
- **GDPR**: Data protection by design, audit trails, right to erasure
- **HIPAA**: Technical safeguards, audit controls, encryption
- **ISO 27001**: Information security management

---

## Quick Start - Secure Deployment

### Docker Deployment (Recommended)
```bash
git clone https://github.com/joecupano/RFexpert-AI/
cd RFexpert-AI
./setup.sh docker

# Interactive setup will:
# 1. Choose LLM backend (Ollama or LocalAI)
# 2. Generate secure passwords
# 3. Deploy with all security features
```

**Access:** `http://localhost:8501`

**Security Verification:**
```bash
# Verify audit logging
docker-compose exec rf-expert-ai-app python -c \
  "from src.audit_logger import audit_logger; \
   print(audit_logger.verify_chain_integrity())"

# Check network isolation
docker-compose ps
docker network inspect rfexpert-ai_backend-network
```

---

## Documentation

### Security & Compliance
- **[SECURE_DEPLOYMENT.md](docs/SECURE_DEPLOYMENT.md)** - Complete secure deployment guide
- **[SECURITY_POLICY.md](docs/SECURITY_POLICY.md)** - Security architecture and policies
- **[COMPLIANCE_CHECKLIST.md](docs/COMPLIANCE_CHECKLIST.md)** - Pre-deployment audit checklist

### Installation & Administration
- **[INSTALLATION.md](docs/INSTALLATION.md)** - Installation guide with multiple deployment options
- **[ADMINISTRATION.md](docs/ADMINISTRATION.md)** - System administration and monitoring
- **[USERGUIDE.md](docs/USERGUIDE.md)** - End-user guide for using the platform
- **[DEVELOPER.md](docs/DEVELOPER.md)** - Development guide and architecture

---

## Platform Overview

### Dual LLM Backend Support

**Choose your on-premise LLM backend:**

#### Ollama (Default)
- Easy setup and configuration
- Great for quick deployment
- Built-in model management
- Active community support

#### LocalAI
- OpenAI-compatible API
- Broader model support
- GPU acceleration
- Multiple model formats

### Key Capabilities

- **Multi-Domain Support**: Create unlimited expert systems, each specialized for different domains
- **Dual Architecture**: Choose between RAG (quick setup) or Fine-tuning (deep specialization)
- **Knowledge Management**: Upload documents, scrape websites, build comprehensive knowledge bases
- **Intelligent Responses**: Get accurate, source-backed answers from your domain expertise
- **⚡ Production Ready**: Complete deployment infrastructure with monitoring and security
- **Audit Trail**: Complete chain of custody for all operations


### **Domain Catalog System**
- Create domain-specific expert systems
- Organize knowledge bases by subject area
- Switch between domains seamlessly
- Isolated data storage per domain

### **Dual Architecture Support**

#### RAG System (Immediate Deployment)
- **Fast Setup**: Ready in 5 minutes
- **Smart Document Processing**: Multi-format support (PDF, DOCX, TXT)
- **Intelligent Caching**: 75% faster responses
- **Web Scraping**: Automated content collection
- **Conversation Memory**: Context-aware discussions
- **Performance Analytics**: Real-time monitoring

#### Fine-Tuning System (Specialized Models)
- **Custom Model Training**: Domain-specific fine-tuning
- **LoRA Integration**: Memory-efficient training
- **Automated Dataset Creation**: Smart Q&A generation
- **Training Script Generation**: Complete training pipeline
- **Model Management**: Version control and deployment

### **Supported Domains**
- **Medical Diagnostics**: Symptoms, treatments, medical knowledge
- **Legal Research**: Case law, regulations, legal analysis
- **RF Communications**: RF theory, antenna design, signal processing
- **Amateur Radio**: Ham radio, licensing, technical reference
- **Scientific Research**: Papers, methodologies, data analysis
- **Engineering**: Technical specifications, design principles
- **Financial Analysis**: Market data, investment strategies
- **Education**: Curriculum content, teaching materials
- **Any Specialized Field**: Completely customizable

---

## Installation

For complete installation instructions, see **[INSTALLATION.md](docs/INSTALLATION.md)**.

### Quick Installation Options

```bash
# Standard installation with services
./setup.sh install

# Docker deployment
./setup.sh docker

# Development setup
./setup.sh dev

# Production with SSL and monitoring
./setup.sh production
```

---

## Getting Started

For detailed usage instructions, see **[USERGUIDE.md](docs/USERGUIDE.md)**.

### Quick Example: RF SIGINT Expert System

1. Create a new domain for RF SIGINT Analysis
2. Add keywords: signals intelligence, RF analysis, signal detection
3. Upload technical documents or scrape websites
4. Choose RAG architecture for quick setup
5. Start asking questions: "How do I detect spread spectrum signals?"

### Quick Example: Amateur Radio Technical Reference

1. Create domain: Amateur Radio Technical Reference
2. Add keywords: ham radio, antenna, propagation, modulation
3. Upload ARRL handbooks and FCC regulations
4. Select fine-tuning for deep expertise
5. Train custom amateur radio expert model

---

## Administration

For complete administration details, see **[ADMINISTRATION.md](docs/ADMINISTRATION.md)**.

### Quick Commands

```bash
# Service management (Standard installation)
sudo systemctl status expert-system
sudo systemctl restart expert-system
sudo journalctl -u expert-system -f

# Docker management
docker-compose ps
docker-compose restart
docker-compose logs -f

# Backups
sudo /opt/expert-system/scripts/backup.sh
```

---

## Contributing

We welcome contributions! For development setup and guidelines, see **[DEVELOPER.md](docs/DEVELOPER.md)**.

```bash
./setup.sh dev
source venv/bin/activate
# Make your changes and submit a pull request
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Streamlit** - Web application framework
- **ChromaDB** - Vector database
- **Ollama** - Local LLM serving
- **Trafilatura** - Web content extraction
- **Llama** - Language models

---

## Support

- **User Guide**: [USERGUIDE.md](docs/USERGUIDE.md)
- **Installation**: [INSTALLATION.md](docs/INSTALLATION.md)
- **Administration**: [ADMINISTRATION.md](docs/ADMINISTRATION.md)
- **Development**: [DEVELOPER.md](docs/DEVELOPER.md)
- **System Status**: `./setup.sh info`

---

**Build intelligent expert systems for any domain. Start your AI transformation today!** 