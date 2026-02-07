# Secure Deployment Guide for Highly Regulated Environments
# ExpertSystem Platform

## Overview

This guide covers deploying the RF Expert AI Platform in highly regulated industries including:
- Financial Services (SOC 2, PCI-DSS)
- Healthcare (HIPAA)
- Government and Defense (ITAR, FedRAMP)
- European organizations (GDPR)

## Key Security Features

### 1. Network Isolation
- **Frontend Network**: Public-facing services only (Nginx, Grafana)
- **Backend Network**: Internal services with no external internet access
- Segmented using Docker bridge networks with IP isolation

### 2. Chain of Custody
- Blockchain-style audit event chaining
- Immutable audit logs with cryptographic hashing
- Complete tracking of all user actions and data access
- 7-year default retention for compliance

### 3. Container Security
- Read-only root filesystems
- Dropped Linux capabilities (cap_drop: ALL)
- Non-root users
- Resource limits (CPU, memory)
- Security contexts (no-new-privileges)

### 4. LLM Backend Options
Choose between two on-premise LLM backends:
- **Ollama**: Easy to use, great for quick deployment
- **LocalAI**: OpenAI-compatible API, broader model support

## Quick Start - Secure Deployment

### Prerequisites

- Ubuntu 24.04 LTS (or compatible Linux)
- Docker 24.0+ and Docker Compose 2.0+
- Minimum 8GB RAM, 4 CPU cores
- 50GB+ free disk space

### Step 1: Clone and Prepare

```bash
git clone <repository-url>
cd RFexpert-AI

# Ensure setup script is executable
chmod +x setup.sh
```

### Step 2: Run Secure Deployment

```bash
# Interactive mode with security features
./setup.sh docker

# You will be prompted to:
# 1. Choose LLM backend (Ollama or LocalAI)
# 2. Automatic secure password generation
# 3. Docker deployment with all security features
```

### Step 3: Post-Deployment Configuration

```bash
# Review generated .env file and customize if needed
nano .env

# Verify all containers are running
docker-compose ps

# Check audit database is ready
docker-compose logs audit-db

# Verify chain of custody
docker-compose exec rf-expert-ai-app python -c "
from src.audit_logger import audit_logger
print(audit_logger.verify_chain_integrity())
"
```

### Step 4: Access the Platform

```bash
# Application
http://localhost:8501

# Grafana (monitoring)
http://localhost:3000
Username: admin
Password: (check .env file for GRAFANA_PASSWORD)
```

## Manual Deployment (Production)

### 1. Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Generate secure passwords
AUDIT_PASS=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
APP_PASS=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
GRAFANA_PASS=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Update .env file
sed -i "s/changeme_secure_password_here/$AUDIT_PASS/" .env
sed -i "s/changeme_app_password_here/$APP_PASS/" .env
sed -i "s/changeme_grafana_password/$GRAFANA_PASS/" .env

# Choose LLM backend
sed -i "s/^LLM_BACKEND=.*/LLM_BACKEND=ollama/" .env  # or localai

# Set compliance framework
sed -i "s/^COMPLIANCE_FRAMEWORK=.*/COMPLIANCE_FRAMEWORK=SOC2/" .env

# Enable security features
sed -i "s/^ENABLE_AUDIT_LOGGING=.*/ENABLE_AUDIT_LOGGING=true/" .env
sed -i "s/^ENABLE_DLP=.*/ENABLE_DLP=true/" .env
```

### 2. Initialize Audit Database

```bash
# Start audit database first
docker-compose up -d audit-db

# Wait for initialization
sleep 10

# Verify audit database
docker-compose exec audit-db psql -U audit_user -d audit_db -c "\dt"

# You should see tables: audit_events, model_interactions, document_access, etc.
```

### 3. Deploy Services

```bash
# Set Docker Compose profile for chosen LLM backend
export COMPOSE_PROFILES=ollama  # or localai

# Deploy with selected backend
docker-compose up -d ollama rf-expert-ai audit-db

# Optional: Deploy monitoring stack
docker-compose up -d prometheus grafana

# Optional: Deploy reverse proxy
docker-compose up -d nginx
```

### 4. SSL/TLS Configuration (Production)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate (assuming nginx is running)
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

## LLM Backend Configuration

### Ollama Setup

```bash
# Ollama is default and automatically configured
# Pull additional models:
docker-compose exec ollama ollama pull mistral
docker-compose exec ollama ollama pull codellama
docker-compose exec ollama ollama list
```

### LocalAI Setup

```bash
# Switch to LocalAI
export COMPOSE_PROFILES=localai
docker-compose up -d localai

# Download models to localai_models volume
docker-compose exec localai wget https://gpt4all.io/models/ggml-gpt4all-j.bin \
  -O /models/ggml-gpt4all-j.bin

# Verify
docker-compose exec localai ls -lh /models/
```

### Switching Between Backends

```bash
# Stop current backend
docker-compose down ollama  # or localai

# Update .env
sed -i "s/^LLM_BACKEND=.*/LLM_BACKEND=localai/" .env  # or ollama

# Start new backend
export COMPOSE_PROFILES=localai  # or ollama
docker-compose up -d localai rf-expert-ai
```

## Audit and Compliance

### Viewing Audit Logs

```bash
# Connect to audit database
docker-compose exec audit-db psql -U audit_reader -d audit_db

# Recent events
SELECT event_type, action, status, timestamp, user_id 
FROM audit_events 
ORDER BY timestamp DESC 
LIMIT 20;

# Security events
SELECT * FROM recent_security_events;

# User activity
SELECT * FROM user_activity_summary;

# Model interactions
SELECT session_id, model_name, status, timestamp 
FROM model_interactions 
ORDER BY timestamp DESC 
LIMIT 10;
```

### Verify Chain Integrity

```bash
# Using Python API
docker-compose exec rf-expert-ai-app python3 << 'EOF'
from src.audit_logger import audit_logger
result = audit_logger.verify_chain_integrity(limit=1000)
print(f"Chain Status: {result['status']}")
print(f"Verified: {result['verified']}")
print(f"Events Checked: {result['events_checked']}")
if 'broken_links' in result and result['broken_links']:
    print(f"Broken Links: {result['broken_links']}")
EOF
```

### Export Audit Report

```bash
# Export to CSV
docker-compose exec audit-db psql -U audit_reader -d audit_db -c \
  "COPY (SELECT * FROM audit_events WHERE timestamp > NOW() - INTERVAL '30 days') 
   TO STDOUT WITH CSV HEADER" > audit_report_30days.csv

# User activity report
docker-compose exec audit-db psql -U audit_reader -d audit_db -c \
  "COPY (SELECT * FROM user_activity_summary) 
   TO STDOUT WITH CSV HEADER" > user_activity.csv
```

## Security Hardening

### 1. Firewall Configuration

```bash
# Using UFW
sudo ufw --force enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS (if using Nginx)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow Streamlit (if direct access needed)
sudo ufw allow 8501/tcp

# Verify
sudo ufw status verbose
```

### 2. Fail2Ban Setup

```bash
# Install
sudo apt install fail2ban

# Configure for Docker
sudo tee /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log

[nginx-limit-req]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
EOF

sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 3. AppArmor Profiles (Optional)

```bash
# Install AppArmor utilities
sudo apt install apparmor-utils

# Create profiles for containers
sudo aa-genprof docker

# Apply profiles
sudo aa-enforce /etc/apparmor.d/docker
```

### 4. Regular Security Updates

```bash
# Create update script
cat > /opt/security-updates.sh << 'EOF'
#!/bin/bash
apt update
apt upgrade -y
apt autoremove -y

# Update Docker images
cd /home/pi/source/RFexpert-AI
docker-compose pull
docker-compose up -d
EOF

chmod +x /opt/security-updates.sh

# Schedule weekly updates (Sundays at 2 AM)
echo "0 2 * * 0 root /opt/security-updates.sh >> /var/log/security-updates.log 2>&1" \
  | sudo tee /etc/cron.d/security-updates
```

## Backup and Recovery

### Automated Backups

```bash
# Create backup script
cat > /opt/backup-rfexpert.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups/rfexpert"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

cd /home/pi/source/RFexpert-AI

# Backup volumes
docker run --rm \
  -v rfexpert-ai_postgres_data:/data \
  -v "$BACKUP_DIR:/backup" \
  alpine tar czf "/backup/postgres_${DATE}.tar.gz" /data

docker run --rm \
  -v rfexpert-ai_audit_db_data:/data \
  -v "$BACKUP_DIR:/backup" \
  alpine tar czf "/backup/audit_db_${DATE}.tar.gz" /data

# Backup configuration
tar czf "$BACKUP_DIR/config_${DATE}.tar.gz" .env docker-compose.yml

# Remove old backups (>90 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +90 -delete

# Encrypt backups (optional)
# gpg --encrypt --recipient your-key-id "$BACKUP_DIR/postgres_${DATE}.tar.gz"

echo "Backup completed: $DATE"
EOF

chmod +x /opt/backup-rfexpert.sh

# Schedule daily backups (2 AM)
echo "0 2 * * * root /opt/backup-rfexpert.sh >> /var/log/backup-rfexpert.log 2>&1" \
  | sudo tee /etc/cron.d/backup-rfexpert
```

### Restore Procedure

```bash
# Stop services
cd /home/pi/source/RFexpert-AI
docker-compose down

# Restore volumes
docker run --rm \
  -v rfexpert-ai_postgres_data:/data \
  -v /opt/backups/rfexpert:/backup \
  alpine sh -c "cd /data && tar xzf /backup/postgres_YYYYMMDD_HHMMSS.tar.gz --strip-components=1"

docker run --rm \
  -v rfexpert-ai_audit_db_data:/data \
  -v /opt/backups/rfexpert:/backup \
  alpine sh -c "cd /data && tar xzf /backup/audit_db_YYYYMMDD_HHMMSS.tar.gz --strip-components=1"

# Restore configuration
tar xzf /opt/backups/rfexpert/config_YYYYMMDD_HHMMSS.tar.gz

# Restart services
docker-compose up -d
```

## Monitoring and Alerting

### Access Grafana

```bash
# URL: http://localhost:3000
# Username: admin
# Password: (from .env file)

# Import dashboards
# - Container metrics
# - Application metrics
# - Security events
# - Audit statistics
```

### Custom Alerts

```bash
# Edit prometheus alerts
nano monitoring/prometheus.yml

# Add alert rules
cat >> monitoring/prometheus.yml << 'EOF'
  - alert: HighFailedAuthentications
    expr: rate(failed_auth_total[5m]) > 5
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: High failed authentication rate
EOF

# Reload Prometheus
docker-compose exec prometheus kill -HUP 1
```

## Troubleshooting

### Check Service Status

```bash
# All services
docker-compose ps

# Logs for specific service
docker-compose logs -f rf-expert-ai
docker-compose logs -f audit-db

# Resource usage
docker stats
```

### Audit Database Issues

```bash
# Connect to audit database
docker-compose exec audit-db psql -U audit_user -d audit_db

# Check table sizes
SELECT 
  schemaname, tablename, 
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

# Check recent events
SELECT COUNT(*), event_type FROM audit_events 
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY event_type;
```

### Network Connectivity

```bash
# Test backend network isolation
docker-compose exec rf-expert-ai-app ping -c 1 8.8.8.8
# Should fail or timeout (no external access)

# Test internal connectivity
docker-compose exec rf-expert-ai-app ping -c 1 audit-db
# Should succeed

# Check network configuration
docker network inspect rfexpert-ai_backend-network
```

## Compliance Verification

Use the provided [COMPLIANCE_CHECKLIST.md](COMPLIANCE_CHECKLIST.md) to verify:

- [ ] All security controls implemented
- [ ] Audit logging capturing all required events
- [ ] Network isolation functioning correctly
- [ ] Chain of custody verified
- [ ] Backup and recovery tested
- [ ] Documentation complete

## Support and Contact

For security issues or compliance questions:
- Email: security@organization.com
- Documentation: [SECURITY_POLICY.md](SECURITY_POLICY.md)
- Issues: GitHub Issues (for non-security issues only)

---

Last Updated: 2026-02-07
Version: 1.0
Classification: INTERNAL
