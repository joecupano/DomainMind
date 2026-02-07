# ExpertSystem Installation Guide

Complete installation guide for Ubuntu 24.04 server deployment with multiple installation methods.

## Related Documentation

- **[README.md](../README.md)** - Project overview and introduction
- **[USERGUIDE.md](USERGUIDE.md)** - End-user guide for using ExpertSystem
- **[ADMINISTRATION.md](ADMINISTRATION.md)** - System administration and troubleshooting
- **[DEVELOPER.md](DEVELOPER.md)** - Development and technical architecture

---

## Quick Start

### Unified Setup Script (Recommended)

All installation methods are now available through a single unified script:

```bash
# Clone or download the project
git clone <your-repo-url>
cd ExpertSystem

# Make setup script executable
chmod +x setup.sh

# Interactive mode - choose your installation method
./setup.sh

# Or use direct commands:
./setup.sh quick       # Quick installation for testing
./setup.sh dev         # Development setup
./setup.sh install     # Standard installation with services
./setup.sh docker      # Docker containerized deployment
./setup.sh production  # Production with SSL and monitoring

# Utility commands:
./setup.sh info        # System information
./setup.sh status      # Service status
./setup.sh update      # Update existing installation
```

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 24.04 LTS (or compatible Linux distribution)
- **RAM**: 4GB (8GB+ recommended)
- **CPU**: 2 cores (4+ cores recommended)
- **Storage**: 20GB free space (50GB+ for production)
- **Network**: Internet connection for initial setup

### Recommended Requirements
- **RAM**: 16GB+ (for fine-tuning capabilities)
- **CPU**: 8+ cores with GPU support for ML training
- **Storage**: 100GB+ SSD
- **GPU**: NVIDIA GPU with CUDA support (optional, for fine-tuning)

## Installation Methods

The unified `setup.sh` script provides five installation methods. Choose based on your needs:

### 1. Quick Install
**Best for:** Testing, evaluation, learning
```bash
./setup.sh quick
```

**Features:**
- Minimal setup (5 minutes)
- Python virtual environment
- No system services
- Local access only
- Includes Ollama installation

**Access:** `http://localhost:8501` via `./start.sh`

### 2. Development Setup
**Best for:** Developers, customization, testing
```bash
./setup.sh dev
```

**Features:**
- Python virtual environment with all dependencies
- ML packages for fine-tuning development
- Hot-reload for code changes
- Local Ollama server
- Development-friendly logging

**Access:** `http://localhost:8501` via `./start-dev.sh`

### 3. Standard Installation
**Best for:** Single-server deployments, internal tools
```bash
./setup.sh install
```

**Features:**
- Systemd service (`expertsystem`) for automatic startup
- Nginx reverse proxy
- System integration
- Service management via systemctl
- Production-ready configuration

**Services installed:**
- `expertsystem` - Main application service
- `ollama` - Local LLM server
- `nginx` - Reverse proxy

**Access:** `http://your-server-ip`

### 4. Docker Deployment
**Best for:** Containerized environments, cloud platforms
```bash
./setup.sh docker
```

**Features:**
- Docker Compose orchestration
- Isolated container environment
- Easy scaling and updates
- Container health monitoring
- Volume persistence

**Services:**
- Application container
- Ollama container
- Nginx proxy (optional)

**Access:** `http://localhost:8080`

### 5. Production Deployment
**Best for:** Public-facing deployments, enterprise use
```bash
./setup.sh production
```

**Interactive prompts for:**
- Domain name configuration
- SSL certificate setup (Let's Encrypt)
- Monitoring stack (Prometheus + Grafana)
- Security hardening

**Features:**
- SSL/TLS certificates with auto-renewal
- Optional monitoring (Prometheus, Grafana)
- Security hardening (fail2ban, UFW firewall)
- Automated daily backups
- Performance optimization
- Log rotation

**Access:** `https://yourdomain.com`

## Dependencies

### Core Python Packages
```bash
pip install streamlit chromadb requests trafilatura PyPDF2 pdfplumber python-docx olefile scikit-learn numpy
```

### Optional ML Packages (Enhanced Features)
```bash
pip install sentence-transformers torch transformers datasets accelerate peft bitsandbytes scipy
```

### System Packages (Ubuntu)
```bash
sudo apt install python3 python3-pip python3-venv build-essential curl git nginx supervisor sqlite3
```

### Ollama (LLM Server)
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama2
```

## Configuration

### Environment Variables
- `LLAMA_MODEL`: Specify Llama model (default: "llama2")
- `OLLAMA_HOST`: Ollama server endpoint (default: "http://localhost:11434")

### Configuration Files
- `.streamlit/config.toml` - Streamlit server configuration
- `data/domains/catalog.json` - Domain catalog metadata
- `nginx.conf` - Web server configuration (if using nginx)

### Directory Structure
```
/opt/expertsystem/                 # Installation directory (standard/production)
├── src/                         # Source code
│   ├── app.py                  # Main application entry point
│   ├── domain_catalog.py       # Domain management
│   ├── generic_rag_system.py   # RAG implementation
│   ├── generic_finetune_system.py # Fine-tuning system
│   └── .streamlit/             # Streamlit configuration
├── data/
│   ├── domains/                # Domain-specific data
│   │   ├── catalog.json       # Domain catalog
│   │   ├── rf_communications/ # Example domain
│   │   └── medical_diagnostics/ # Additional domains
│   └── cache/                  # System caches
├── logs/                       # Application logs
├── backups/                    # Automatic backups (production)
└── scripts/                    # Management scripts (production)
```

## Service Management

For complete service management details, see **[ADMINISTRATION.md](ADMINISTRATION.md)**.

### Quick Reference

**Standard/Production Installation:**
```bash
# Service control
sudo systemctl start expertsystem
sudo systemctl stop expertsystem
sudo systemctl restart expertsystem
sudo systemctl status expertsystem

# View logs
sudo journalctl -u expertsystem -f
```

**Docker Deployment:**
```bash
# Container control
docker-compose up -d
docker-compose down
docker-compose restart
docker-compose ps

# View logs
docker-compose logs -f
```

**Development/Quick Install:**
```bash
# Activate environment and start
source venv/bin/activate
streamlit run src/app.py

# Or use the provided script
./start.sh          # Quick install
./start-dev.sh      # Development
```

**Setup Script Management:**
```bash
./setup.sh status   # Check service status
./setup.sh update   # Update installation
./setup.sh info     # System information
```

## Troubleshooting

For comprehensive troubleshooting, see **[ADMINISTRATION.md](ADMINISTRATION.md)**.

### Quick Diagnostics

```bash
# Check system information
./setup.sh info

# Check service status
./setup.sh status

# Update existing installation
./setup.sh update

# Start fresh with development mode if issues
./setup.sh dev
```

### Common Issues

**Service won't start:**
```bash
sudo journalctl -u expertsystem -n 50  # View recent logs
./setup.sh status                     # Check status
```

**Port already in use:**
```bash
sudo lsof -i :8501    # Check what's using port 8501
sudo lsof -i :80      # Check what's using port 80
```

**Ollama not responding:**
```bash
sudo systemctl status ollama
sudo systemctl restart ollama
ollama list           # Check installed models
```

For detailed troubleshooting, service management, monitoring, backups, security configuration, and performance tuning, see **[ADMINISTRATION.md](ADMINISTRATION.md)**.

---

## Next Steps

After installation:

1. **Access the application** at your configured URL
2. **Read the User Guide**: See [USERGUIDE.md](USERGUIDE.md) to create your first domain
3. **Configure administration**: See [ADMINISTRATION.md](ADMINISTRATION.md) for system management
4. **For developers**: See [DEVELOPER.md](DEVELOPER.md) for technical details

---