# Security Quick Start Guide
# ExpertSystem Platform - Secure Deployment in 5 Minutes

## Prerequisites

- Ubuntu 24.04 LTS (or compatible)
- Docker and Docker Compose installed
- 8GB RAM minimum
- 50GB free disk space

## 1. Clone and Setup (1 minute)

```bash
# Clone repository
git clone <your-repo-url>
cd RFexpert-AI

# Make setup script executable
chmod +x setup.sh
```

## 2. Run Secure Deployment (3 minutes)

```bash
# Interactive deployment with security features
./setup.sh docker
```

**You will be prompted for:**
1. **LLM Backend Choice**
   - Option 1: Ollama (recommended for quick start)
   - Option 2: LocalAI (for OpenAI-compatible API)

The script will automatically:
- Generate strong random passwords
- Create .env file with security settings
- Deploy containers with isolation
- Initialize audit database
- Start all services

## 3. Verify Deployment (1 minute)

```bash
# Check all containers are running
docker-compose ps

# Should see:
# - rf-expert-ai-app (main application)
# - ollama or localai (LLM backend)
# - audit-db (chain of custody database)

# Verify audit logging
docker-compose exec rf-expert-ai-app python3 -c \
  "from src.audit_logger import audit_logger; \
   print('Audit logging:', 'ENABLED' if audit_logger.enabled else 'DISABLED'); \
   print('Chain integrity:', audit_logger.verify_chain_integrity())"
```

## 4. Access the Platform

```bash
# Main application
http://localhost:8501

# Grafana (monitoring) - optional
http://localhost:3000
Username: admin
Password: (check .env file for GRAFANA_PASSWORD)
```

## 5. Security Verification Checklist

```bash
# Network isolation
docker network inspect rfexpert-ai_backend-network | grep internal
# Should show: "internal": true

# Audit database initialized
docker-compose exec audit-db psql -U audit_user -d audit_db -c "\dt"
# Should list: audit_events, model_interactions, document_access, etc.

# No root containers
docker-compose exec rf-expert-ai-app whoami
# Should NOT be "root"

# Read-only filesystem (main app)
docker inspect rf-expert-ai-app | grep ReadonlyRootfs
# Should show: "ReadonlyRootfs": true

# Strong passwords generated
cat .env | grep PASSWORD
# Should show random 25-character passwords
```

## Key Security Features Enabled

### Network Isolation
- **Frontend network** (172.20.0.0/24): Public services only
- **Backend network** (172.21.0.0/24): Internal only, no internet access
- Services communicate through defined connections only

### Chain of Custody
- Every action logged with timestamp, user, session, IP
- Blockchain-style event chaining (each event linked to previous)
- Immutable logs (deletion prevented by database triggers)
- 7-year retention for compliance

### Container Security
- Read-only root filesystems
- Dropped all Linux capabilities, only essential ones added back
- Non-root users in all containers
- Resource limits (CPU, memory) enforced
- Security contexts (no-new-privileges)

### Data Protection
- Documents mounted read-only where possible
- Separate audit log volume
- Hash verification for all files
- Access tracking in audit database

## Quick Commands

### View Audit Logs
```bash
# Connect to audit database
docker-compose exec audit-db psql -U audit_reader -d audit_db

# Recent events
SELECT event_type, action, status, timestamp FROM audit_events ORDER BY timestamp DESC LIMIT 10;

# User activity
SELECT * FROM user_activity_summary;

# Security events
SELECT * FROM recent_security_events;
```

### Switch LLM Backend
```bash
# Stop current backend
docker-compose down ollama  # or localai

# Update .env
nano .env
# Change: LLM_BACKEND=localai  # or ollama

# Start new backend
export COMPOSE_PROFILES=localai  # or ollama
docker-compose up -d localai rf-expert-ai
```

### View Logs
```bash
# Application logs
docker-compose logs -f rf-expert-ai

# Audit database logs
docker-compose logs -f audit-db

# LLM backend logs
docker-compose logs -f ollama  # or localai
```

### Backup
```bash
# Quick backup
docker run --rm \
  -v rfexpert-ai_audit_db_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/audit-backup-$(date +%Y%m%d).tar.gz /data
```

### Stop/Start
```bash
# Stop all
docker-compose down

# Start all
export COMPOSE_PROFILES=ollama  # or localai
docker-compose up -d

# Restart specific service
docker-compose restart rf-expert-ai
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs rf-expert-ai-app

# Common issues:
# - Audit database not ready: wait 10 seconds and restart
# - Port conflicts: check if 8501 already in use
```

### Audit logging not working
```bash
# Check audit database connection
docker-compose exec audit-db pg_isready

# Check environment variable
docker-compose exec rf-expert-ai-app env | grep ENABLE_AUDIT_LOGGING
# Should show: ENABLE_AUDIT_LOGGING=true

# Reinitialize if needed
docker-compose down audit-db
docker volume rm rfexpert-ai_audit_db_data
docker-compose up -d audit-db
```

### Network isolation test
```bash
# Backend should NOT reach internet
docker-compose exec rf-expert-ai-app ping -c 1 8.8.8.8
# Should fail or timeout

# But should reach audit-db
docker-compose exec rf-expert-ai-app ping -c 1 audit-db
# Should succeed
```

## Next Steps

1. **Customize Configuration**: Edit `.env` file for your requirements
2. **Enable SSL**: Follow [SECURE_DEPLOYMENT.md](SECURE_DEPLOYMENT.md) for SSL setup
3. **Configure Monitoring**: Set up Grafana dashboards
4. **Compliance Check**: Use [COMPLIANCE_CHECKLIST.md](COMPLIANCE_CHECKLIST.md)
5. **Security Hardening**: See [SECURITY_POLICY.md](SECURITY_POLICY.md)

## Support

- **Security Issues**: security@organization.com
- **Documentation**: See SECURE_DEPLOYMENT.md for detailed guide
- **Compliance**: See COMPLIANCE_CHECKLIST.md for audit requirements

---

**Important Security Notes:**
- Change default passwords immediately in production
- Enable firewall (UFW) to restrict access
- Set up regular backups
- Review audit logs regularly
- Keep Docker images updated

---

Last Updated: 2026-02-07
Classification: INTERNAL
