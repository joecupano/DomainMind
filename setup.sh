#!/bin/bash

# ExpertSystem Universal Setup Script
# Unified installation, deployment, and management tool for Ubuntu 24.04

set -e

# Script version
VERSION="2.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration variables
INSTALL_DIR="/opt/expertsystem"
DOMAIN_NAME=""
EMAIL=""
ENABLE_SSL=false
ENABLE_MONITORING=false
BACKUP_RETENTION=7
DEPLOYMENT_MODE=""

# Logging functions
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
section() { echo -e "\n${CYAN}═══════════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}═══════════════════════════════════════════${NC}\n"; }

# Display banner
show_banner() {
    clear
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                                                           ║"
    echo "║                 ExpertSystem Platform                     ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Show main menu
show_main_menu() {
    echo -e "${GREEN}Choose Installation Mode:${NC}\n"
    echo "  1)  Quick Install      - Minimal setup for testing"
    echo "  2)  Development        - Local development environment"
    echo "  3)  Standard Install   - Full system with services"
    echo "  4)  Docker Deploy      - Containerized deployment"
    echo "  5)  Production Deploy  - Enterprise with SSL/monitoring"
    echo ""
    echo "  6)  System Info        - Check system status"
    echo "  7)  Update             - Update existing installation"
    echo "  8)  Status             - Service status and health"
    echo "  9)  Exit"
    echo ""
    read -p "Select option [1-9]: " choice
    echo ""
}

# Check system requirements
check_requirements() {
    section "System Requirements Check"
    
    # Check OS
    if ! grep -q "Ubuntu" /etc/os-release 2>/dev/null; then
        warn "This script is optimized for Ubuntu 24.04"
        if ! grep -q "Debian\\|Ubuntu" /etc/os-release 2>/dev/null; then
            error "Unsupported operating system"
            exit 1
        fi
    fi
    
    # Check disk space (need at least 10GB)
    local available_space=$(df / | awk 'NR==2 {print $4}')
    local required_space=10485760
    
    if [ "$available_space" -lt "$required_space" ]; then
        error "Insufficient disk space. At least 10GB required."
        exit 1
    fi
    
    # Check RAM
    local total_ram=$(free | awk 'NR==2{print $2}')
    local required_ram=3145728
    
    if [ "$total_ram" -lt "$required_ram" ]; then
        warn "Less than 4GB RAM detected. Performance may be limited."
    fi
    
    # Check CPU cores
    local cpu_cores=$(nproc)
    if [ "$cpu_cores" -lt 2 ]; then
        warn "Single-core system detected. Multi-core recommended."
    fi
    
    success "System requirements check completed"
    echo "  OS: $(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown")"
    echo "  RAM: $(free -h | awk 'NR==2{print $2}')"
    echo "  CPU: $cpu_cores cores"
    echo "  Disk: $(df -h / | awk 'NR==2{print $4}') available"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root. Run as a regular user with sudo privileges."
        exit 1
    fi
    
    if ! sudo -n true 2>/dev/null; then
        log "This script requires sudo privileges. You may be prompted for your password."
    fi
}

# Update system packages
update_system() {
    log "Updating system packages..."
    sudo apt update -qq
    sudo apt upgrade -y -qq
    success "System packages updated"
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    
    sudo apt install -y -qq \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        git \
        curl \
        wget \
        unzip \
        software-properties-common \
        sqlite3 \
        libsqlite3-dev \
        htop \
        tree \
        jq \
        || error "Failed to install system dependencies"
    
    success "System dependencies installed"
}

# Select LLM backend
select_llm_backend() {
    section "LLM Backend Selection"
    
    echo -e "${GREEN}Choose LLM Backend:${NC}\n"
    echo "  1)  Ollama     - Easy to use, great for quick setup"
    echo "  2)  LocalAI    - OpenAI-compatible, more models"
    echo ""
    read -p "Select LLM backend [1-2] (default: 1): " llm_choice
    
    case ${llm_choice:-1} in
        1)
            LLM_BACKEND="ollama"
            log "Selected: Ollama"
            ;;
        2)
            LLM_BACKEND="localai"
            log "Selected: LocalAI"
            ;;
        *)
            LLM_BACKEND="ollama"
            log "Defaulting to: Ollama"
            ;;
    esac
    
    # Create/update .env file
    if [ ! -f .env ]; then
        cp .env.example .env 2>/dev/null || touch .env
    fi
    
    # Update LLM_BACKEND in .env
    if grep -q "LLM_BACKEND=" .env; then
        sed -i "s/^LLM_BACKEND=.*/LLM_BACKEND=$LLM_BACKEND/" .env
    else
        echo "LLM_BACKEND=$LLM_BACKEND" >> .env
    fi
    
    success "LLM backend configured: $LLM_BACKEND"
}

# Install Ollama
install_ollama() {
    if command -v ollama &> /dev/null; then
        success "Ollama already installed"
        return
    fi
    
    log "Installing Ollama..."
    if curl -fsSL https://ollama.ai/install.sh | sh; then
        success "Ollama installed"
        
        # Start Ollama service if systemd is available
        if command -v systemctl &> /dev/null; then
            sudo systemctl enable ollama 2>/dev/null || true
            sudo systemctl start ollama 2>/dev/null || true
        fi
        
        # Pull default model in background
        log "Pulling llama2 model (this may take a while)..."
        (ollama pull llama2 &) || warn "Failed to pull llama2 model. Run 'ollama pull llama2' manually."
    else
        warn "Ollama installation failed. You can install it manually later."
    fi
}

# Install LocalAI
install_localai() {
    log "LocalAI will be installed via Docker"
    log "Pulling LocalAI image..."
    
    if command -v docker &> /dev/null; then
        docker pull quay.io/go-skynet/local-ai:latest || warn "Failed to pull LocalAI image"
        success "LocalAI image ready"
    else
        warn "Docker not available. LocalAI requires Docker."
    fi
}

# Quick installation mode
quick_install() {
    section "Quick Installation"
    
    check_root
    check_requirements
    select_llm_backend
    
    log "Installing Python and basic dependencies..."
    sudo apt update -qq
    sudo apt install -y python3 python3-pip python3-venv git curl
    
    # Create directory
    mkdir -p expertsystem-quick
    cd expertsystem-quick
    
    log "Setting up Python environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
    
    # Install Python packages
    log "Installing Python packages..."
    pip install -q \
        streamlit chromadb requests trafilatura \
        PyPDF2 pdfplumber python-docx olefile \
        scikit-learn numpy psycopg2-binary
    
    # Create directory structure
    mkdir -p data/domains data/cache logs audit_logs src/.streamlit
    
    # Copy source files
    if [ -d "../src" ]; then
        log "Copying source files..."
        cp -r ../src/* src/
    else
        warn "Source files not found. Please ensure you're running from the project directory."
    fi
    
    # Create Streamlit config
    cat > src/.streamlit/config.toml << 'EOF'
[server]
headless = true
address = "0.0.0.0"
port = 8501

[theme]
base = "light"
EOF
    
    # Create startup script
    cat > start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run src/app.py
EOF
    chmod +x start.sh
    
    # Install selected LLM backend
    if [ "$LLM_BACKEND" = "ollama" ]; then
        install_ollama
    else
        install_localai
    fi
    
    success "Quick installation completed!"
    echo ""
    echo "LLM Backend: $LLM_BACKEND"
    echo "Start with: ./start.sh"
    echo "Access at: http://localhost:8501"
}

# Development setup
dev_setup() {
    section "Development Setup"
    
    check_root
    check_requirements
    select_llm_backend
    
    log "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    
    # Install core dependencies
    log "Installing core dependencies..."
    pip install streamlit chromadb requests trafilatura \
        PyPDF2 pdfplumber python-docx olefile \
        scikit-learn numpy psycopg2-binary
    
    # Install optional ML dependencies
    log "Installing ML dependencies (optional)..."
    pip install sentence-transformers torch transformers datasets accelerate peft bitsandbytes scipy \
        || warn "Some ML packages failed (optional for fine-tuning)"
    
    # Create directories
    mkdir -p data/domains data/cache logs audit_logs
    
    # Copy .env.example if .env doesn't exist
    [ ! -f .env ] && cp .env.example .env 2>/dev/null || true
    
    # Create start script
    cat > start-dev.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
EOF
    chmod +x start-dev.sh
    
    # Install selected LLM backend
    if [ "$LLM_BACKEND" = "ollama" ]; then
        install_ollama
    else
        install_localai
    fi
    
    success "Development environment ready!"
    echo ""
    echo "LLM Backend: $LLM_BACKEND"
    echo "Start with: ./start-dev.sh"
    echo "Or: source venv/bin/activate && streamlit run src/app.py"
    echo "Access at: http://localhost:8501"
}

# Standard installation with systemd
standard_install() {
    section "Standard Installation"
    
    check_root
    check_requirements
    update_system
    install_system_deps
    
    # Install nginx and supervisor
    log "Installing web server..."
    sudo apt install -y nginx supervisor
    
    # Create installation directory
    log "Creating installation directory..."
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown $USER:$USER "$INSTALL_DIR"
    
    # Copy files
    log "Copying application files..."
    cp -r src data "$INSTALL_DIR/"
    cp -r nginx monitoring "$INSTALL_DIR/" 2>/dev/null || true
    
    # Setup Python environment
    log "Setting up Python environment..."
    cd "$INSTALL_DIR"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install streamlit chromadb requests trafilatura \
        PyPDF2 pdfplumber python-docx olefile \
        scikit-learn numpy
    
    # Create systemd service
    log "Creating systemd service..."
    sudo tee /etc/systemd/system/expertsystem.service > /dev/null << EOF
[Unit]
Description=ExpertSystem Platform
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin"
ExecStart=$INSTALL_DIR/venv/bin/streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Configure nginx
    log "Configuring nginx..."
    sudo tee /etc/nginx/sites-available/expertsystem > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
EOF
    
    sudo ln -sf /etc/nginx/sites-available/expertsystem /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t
    
    install_ollama
    
    # Enable and start services
    log "Starting services..."
    sudo systemctl daemon-reload
    sudo systemctl enable expertsystem
    sudo systemctl start expertsystem
    sudo systemctl restart nginx
    
    success "Standard installation completed!"
    echo ""
    echo "Service: sudo systemctl status expertsystem"
    echo "Logs: sudo journalctl -u expertsystem -f"
    echo "Access at: http://$(hostname -I | awk '{print $1}')"
}

# Docker deployment
docker_deploy() {
    section "Docker Deployment"
    
    check_root
    check_requirements
    select_llm_backend
    
    # Install Docker
    if ! command -v docker &> /dev/null; then
        log "Installing Docker..."
        curl -fsSL https://get.docker.com | sh
        sudo usermod -aG docker $USER
        success "Docker installed. You may need to log out and back in."
    fi
    
    # Install Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log "Installing Docker Compose..."
        sudo apt install -y docker-compose
    fi
    
    # Check if docker-compose.yml exists
    if [ ! -f "docker-compose.yml" ]; then
        error "docker-compose.yml not found. Please run from project directory."
        exit 1
    fi
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log "Creating .env file from template..."
        cp .env.example .env
        
        # Generate secure passwords
        AUDIT_PASS=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        APP_PASS=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        GRAFANA_PASS=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
        
        sed -i "s/changeme_secure_password_here/$AUDIT_PASS/" .env
        sed -i "s/changeme_app_password_here/$APP_PASS/" .env
        sed -i "s/changeme_grafana_password/$GRAFANA_PASS/" .env
        sed -i "s/^LLM_BACKEND=.*/LLM_BACKEND=$LLM_BACKEND/" .env
        
        success "Generated secure passwords in .env file"
    fi
    
    # Set Docker Compose profile based on LLM backend
    export COMPOSE_PROFILES=$LLM_BACKEND
    
    log "Building and starting containers with $LLM_BACKEND backend..."
    docker-compose up -d --build $LLM_BACKEND rf-expert-ai audit-db
    
    success "Docker deployment completed!"
    echo ""
    echo "LLM Backend: $LLM_BACKEND"
    echo "Status: docker-compose ps"
    echo "Logs: docker-compose logs -f"
    echo "Stop: docker-compose down"
    echo "Access at: http://localhost:8501"
    echo ""
    echo "Security features enabled:"
    echo "  ✓ Network isolation (frontend/backend)"
    echo "  ✓ Audit logging & chain of custody"
    echo "  ✓ Resource limits & quotas"
    echo "  ✓ Read-only filesystems"
    echo "  ✓ Dropped capabilities"
}

# Production deployment
production_deploy() {
    section "Production Deployment"
    
    # Get configuration
    read -p "Domain name (e.g., example.com): " DOMAIN_NAME
    read -p "Email for SSL certificate: " EMAIL
    read -p "Enable monitoring? (y/N): " enable_mon
    
    [[ $enable_mon =~ ^[Yy]$ ]] && ENABLE_MONITORING=true || ENABLE_MONITORING=false
    
    if [ -n "$DOMAIN_NAME" ] && [ -n "$EMAIL" ]; then
        ENABLE_SSL=true
    fi
    
    # Run standard install first
    standard_install
    
    # Install SSL if requested
    if [ "$ENABLE_SSL" = true ]; then
        section "Installing SSL Certificate"
        
        sudo apt install -y certbot python3-certbot-nginx
        sudo certbot --nginx -d "$DOMAIN_NAME" --non-interactive --agree-tos --email "$EMAIL" \
            || warn "SSL installation failed. You can configure it manually later."
        
        # Setup auto-renewal
        echo "0 3 * * * root certbot renew --quiet" | sudo tee /etc/cron.d/certbot-renew > /dev/null
    fi
    
    # Install monitoring if requested
    if [ "$ENABLE_MONITORING" = true ]; then
        section "Installing Monitoring Stack"
        
        # Install Prometheus
        wget -q https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
        tar xzf prometheus-2.45.0.linux-amd64.tar.gz
        sudo mv prometheus-2.45.0.linux-amd64 /opt/prometheus
        rm prometheus-2.45.0.linux-amd64.tar.gz
        
        # Install Grafana
        sudo apt install -y software-properties-common
        sudo add-apt-repository -y "deb https://packages.grafana.com/oss/deb stable main"
        wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
        sudo apt update && sudo apt install -y grafana
        
        sudo systemctl enable grafana-server
        sudo systemctl start grafana-server
        
        success "Monitoring installed. Access Grafana at http://$(hostname -I | awk '{print $1}'):3000"
    fi
    
    # Security hardening
    section "Security Hardening"
    
    # Install fail2ban
    sudo apt install -y fail2ban
    sudo systemctl enable fail2ban
    sudo systemctl start fail2ban
    
    # Configure firewall
    if command -v ufw &> /dev/null; then
        sudo ufw --force enable
        sudo ufw allow ssh
        sudo ufw allow 80/tcp
        sudo ufw allow 443/tcp
        [ "$ENABLE_MONITORING" = true ] && sudo ufw allow 3000/tcp
    fi
    
    # Setup backups
    section "Configuring Backups"
    
    sudo mkdir -p "$INSTALL_DIR/backups"
    
    cat > "$INSTALL_DIR/scripts/backup.sh" << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/expertsystem/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tar -czf "$BACKUP_DIR/backup-$TIMESTAMP.tar.gz" \
    /opt/expertsystem/data \
    /opt/expertsystem/src/.streamlit
find "$BACKUP_DIR" -name "backup-*.tar.gz" -mtime +7 -delete
EOF
    
    chmod +x "$INSTALL_DIR/scripts/backup.sh"
    echo "0 2 * * * root $INSTALL_DIR/scripts/backup.sh" | sudo tee /etc/cron.d/expertsystem-backup > /dev/null
    
    success "Production deployment completed!"
    echo ""
    [ -n "$DOMAIN_NAME" ] && echo "Access at: https://$DOMAIN_NAME" || echo "Access at: http://$(hostname -I | awk '{print $1}')"
}

# Show system information
show_system_info() {
    section "System Information"
    
    echo -e "${CYAN}Operating System:${NC}"
    lsb_release -a 2>/dev/null || cat /etc/os-release
    
    echo -e "\n${CYAN}Hardware:${NC}"
    echo "CPU: $(nproc) cores"
    echo "RAM: $(free -h | awk 'NR==2{print $2}')"
    echo "Disk: $(df -h / | awk 'NR==2{print $4}') available"
    
    echo -e "\n${CYAN}Software:${NC}"
    echo "Python: $(python3 --version 2>&1)"
    echo "Docker: $(docker --version 2>&1 || echo "Not installed")"
    echo "Ollama: $(ollama --version 2>&1 || echo "Not installed")"
    
    if systemctl is-active --quiet expertsystem; then
        echo -e "\n${CYAN}ExpertSystem Status:${NC}"
        systemctl status expertsystem --no-pager | head -10
    fi
}

# Show service status
show_status() {
    section "Service Status"
    
    if systemctl list-units --type=service | grep -q expertsystem; then
        sudo systemctl status expertsystem --no-pager
    else
        warn "ExpertSystem service not found. Is it installed?"
    fi
    
    if command -v docker-compose &> /dev/null && [ -f "docker-compose.yml" ]; then
        echo -e "\n${CYAN}Docker Status:${NC}"
        docker-compose ps
    fi
}

# Update existing installation
update_installation() {
    section "Update Installation"
    
    if [ -d "$INSTALL_DIR" ]; then
        log "Updating standard installation..."
        cd "$INSTALL_DIR"
        source venv/bin/activate
        pip install --upgrade streamlit chromadb requests trafilatura \
            PyPDF2 pdfplumber python-docx olefile \
            scikit-learn numpy
        sudo systemctl restart expertsystem
        success "Update completed"
    elif [ -f "docker-compose.yml" ]; then
        log "Updating Docker deployment..."
        docker-compose pull
        docker-compose up -d --build
        success "Update completed"
    else
        warn "No installation found to update"
    fi
}

# Main execution
main() {
    show_banner
    
    # If arguments provided, run non-interactively
    if [ $# -gt 0 ]; then
        case "$1" in
            quick) quick_install ;;
            dev) dev_setup ;;
            install) standard_install ;;
            docker) docker_deploy ;;
            production) production_deploy ;;
            info) show_system_info ;;
            status) show_status ;;
            update) update_installation ;;
            *)
                error "Unknown command: $1"
                echo "Usage: $0 {quick|dev|install|docker|production|info|status|update}"
                exit 1
                ;;
        esac
        exit 0
    fi
    
    # Interactive mode
    while true; do
        show_main_menu
        
        case $choice in
            1) quick_install; break ;;
            2) dev_setup; break ;;
            3) standard_install; break ;;
            4) docker_deploy; break ;;
            5) production_deploy; break ;;
            6) show_system_info; read -p "Press Enter to continue..."; continue ;;
            7) update_installation; break ;;
            8) show_status; read -p "Press Enter to continue..."; continue ;;
            9) log "Exiting..."; exit 0 ;;
            *) error "Invalid option"; sleep 2; continue ;;
        esac
    done
    
    echo ""
    log "Setup complete! "
}

# Run main function
main "$@"
