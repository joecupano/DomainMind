# ExpertSystem Administration Guide

System administration guide for managing ExpertSystem deployments, including service management, monitoring, backups, security, and troubleshooting.

## Table of Contents

- [Service Management](#service-management)
- [Monitoring and Logs](#monitoring-and-logs)
- [Backup and Restore](#backup-and-restore)
- [Security Configuration](#security-configuration)
- [Performance Tuning](#performance-tuning)
- [Updates and Maintenance](#updates-and-maintenance)
- [Troubleshooting](#troubleshooting)

---

## Service Management

### Standard Installation (Systemd)

ExpertSystem runs as a systemd service for automatic startup and management.

#### Service Control

```bash
# Start the service
sudo systemctl start expert-system

# Stop the service
sudo systemctl stop expert-system

# Restart the service
sudo systemctl restart expert-system

# Check service status
sudo systemctl status expert-system

# Enable auto-start on boot
sudo systemctl enable expert-system

# Disable auto-start
sudo systemctl disable expert-system
```

#### Ollama Service

```bash
# Ollama LLM server management
sudo systemctl start ollama
sudo systemctl stop ollama
sudo systemctl restart ollama
sudo systemctl status ollama
```

#### Nginx Service

```bash
# Web server / reverse proxy
sudo systemctl start nginx
sudo systemctl stop nginx
sudo systemctl restart nginx
sudo systemctl status nginx

# Test configuration
sudo nginx -t

# Reload configuration without downtime
sudo systemctl reload nginx
```

### Docker Deployment

#### Container Management

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart services
docker-compose restart

# View running containers
docker-compose ps

# View resource usage
docker stats
```

#### Individual Container Control

```bash
# Restart specific service
docker-compose restart expert-system-app

# View logs for specific service
docker-compose logs -f expert-system-app

# Execute command in container
docker-compose exec expert-system-app bash
```

#### Container Maintenance

```bash
# Rebuild containers
docker-compose down
docker-compose up -d --build

# Remove unused images and volumes
docker system prune -a
docker volume prune

# Update images
docker-compose pull
docker-compose up -d
```

### Management Scripts

Convenience scripts for common tasks:

```bash
# Located in /opt/expert-system/scripts/

# Start application
./scripts/start.sh

# Stop application
./scripts/stop.sh

# Restart application
./scripts/restart.sh

# Check status
./scripts/status.sh

# View logs
./scripts/logs.sh

# Create backup
./scripts/backup.sh

# Update application
./scripts/update.sh
```

---

## 📊 Monitoring and Logs

### Application Logs

#### Systemd Service Logs

```bash
# View recent logs
sudo journalctl -u expert-system

# Follow logs in real-time
sudo journalctl -u expert-system -f

# View logs from today
sudo journalctl -u expert-system --since today

# View last 100 lines
sudo journalctl -u expert-system -n 100

# View logs with timestamps
sudo journalctl -u expert-system -o verbose
```

#### Docker Logs

```bash
# View logs for all services
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# View logs for specific service
docker-compose logs expert-system-app

# Tail last 100 lines
docker-compose logs --tail=100
```

#### Application Log Files

```bash
# Standard installation
tail -f /opt/expert-system/logs/app.log
tail -f /opt/expert-system/logs/error.log

# Check log directory
ls -lah /opt/expert-system/logs/
```

### Web Server Logs

```bash
# Nginx access logs
sudo tail -f /var/log/nginx/access.log

# Nginx error logs
sudo tail -f /var/log/nginx/error.log

# Search for errors
sudo grep "error" /var/log/nginx/error.log

# View by date
sudo cat /var/log/nginx/access.log | grep "07/Feb/2026"
```

### System Monitoring

#### Resource Usage

```bash
# Real-time system monitor
htop

# Memory usage
free -h

# Disk usage
df -h

# Disk usage by directory
du -sh /opt/expert-system/*

# Check specific domain data
du -sh /opt/expert-system/data/domains/*
```

#### Process Monitoring

```bash
# Find ExpertSystem processes
ps aux | grep streamlit
ps aux | grep ollama

# Monitor CPU and memory usage
top -p $(pgrep -d',' streamlit)

# Check network connections
sudo netstat -tulnp | grep 8501
sudo ss -tulnp | grep 8501
```

### Production Monitoring Stack

If deployed with monitoring (via `./setup.sh production`):

#### Grafana Dashboard

- **URL**: `http://your-server-ip:3000`
- **Default credentials**: admin/admin (change on first login)
- **Features**:
  - System resource metrics
  - Application performance
  - Custom domain dashboards
  - Alerting configuration

#### Prometheus Metrics

- **URL**: `http://your-server-ip:9090`
- **Metrics available**:
  - CPU, memory, disk usage
  - Application response times
  - Query performance
  - Cache hit rates

#### Setting Up Alerts

```bash
# Edit Prometheus alert rules
sudo nano /opt/expert-system/monitoring/alert.rules.yml

# Restart Prometheus
sudo systemctl restart prometheus

# Test alerts
curl -X POST http://localhost:9090/-/reload
```

---

## 💾 Backup and Restore

### Automatic Backups

Production deployments include automatic daily backups.

#### Configuration

```bash
# Backup script location
/opt/expert-system/scripts/backup.sh

# Cron job (daily at 2 AM)
sudo crontab -l | grep backup

# Edit backup schedule
sudo crontab -e
```

#### Backup Settings

```bash
# Edit backup configuration
sudo nano /opt/expert-system/scripts/backup.sh

# Key settings:
BACKUP_DIR="/opt/expert-system/backups"
RETENTION_DAYS=7  # Keep backups for 7 days
```

### Manual Backups

#### Full System Backup

```bash
# Standard installation
sudo /opt/expert-system/scripts/backup.sh

# Docker deployment
./scripts/docker-backup.sh

# Verify backup created
ls -lh /opt/expert-system/backups/
```

#### Selective Backups

```bash
# Backup specific domain
tar -czf domain-backup.tar.gz /opt/expert-system/data/domains/rf_communications/

# Backup all domains
tar -czf domains-backup.tar.gz /opt/expert-system/data/domains/

# Backup configuration
tar -czf config-backup.tar.gz /opt/expert-system/*.py /opt/expert-system/config.py

# Backup database only
tar -czf db-backup.tar.gz /opt/expert-system/data/domains/*/vector_db/
```

### Restore from Backup

#### Full Restore

```bash
# Stop services
sudo systemctl stop expert-system

# Navigate to installation directory
cd /opt/expert-system

# Extract backup (CAREFUL: overwrites existing data)
sudo tar -xzf backups/expert-system-backup-20260207_020000.tar.gz

# Set correct permissions
sudo chown -R expert-system:expert-system /opt/expert-system/

# Start services
sudo systemctl start expert-system
```

#### Restore Specific Domain

```bash
# Stop services
sudo systemctl stop expert-system

# Extract domain backup
sudo tar -xzf domain-backup.tar.gz -C /opt/expert-system/data/domains/

# Fix permissions
sudo chown -R expert-system:expert-system /opt/expert-system/data/domains/

# Start services
sudo systemctl start expert-system
```

#### Docker Restore

```bash
# Stop containers
docker-compose down

# Restore data volume
docker run --rm -v expert-system_data:/data -v $(pwd)/backups:/backups alpine tar -xzf /backups/backup.tar.gz -C /data

# Start containers
docker-compose up -d
```

### Backup Best Practices

1. **Regular Backups**: Daily automatic backups minimum
2. **Off-site Storage**: Copy backups to remote location
3. **Test Restores**: Verify backups work periodically
4. **Document Process**: Keep restore procedures updated
5. **Monitor Space**: Ensure sufficient disk space for backups

```bash
# Copy backups to remote server
rsync -avz /opt/expert-system/backups/ user@backup-server:/backups/expertsystem/

# Automated remote backup (add to cron)
0 3 * * * rsync -avz /opt/expert-system/backups/ user@backup-server:/backups/expertsystem/
```

---

## 🔐 Security Configuration

### Firewall Setup

#### UFW (Uncomplicated Firewall)

```bash
# Install UFW
sudo apt install ufw

# Allow SSH (important - do this first!)
sudo ufw allow ssh

# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow specific application port if needed
sudo ufw allow 8501/tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status verbose

# Remove rule
sudo ufw delete allow 8501/tcp
```

### SSL/TLS Configuration

#### Automatic SSL (Let's Encrypt)

```bash
# Deploy with automatic SSL (uses interactive prompts)
./setup.sh production

# Manual certificate installation
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Test auto-renewal
sudo certbot renew --dry-run

# Force renewal
sudo certbot renew --force-renewal
```

#### Certificate Management

```bash
# List certificates
sudo certbot certificates

# Revoke certificate
sudo certbot revoke --cert-path /etc/letsencrypt/live/yourdomain.com/cert.pem

# Delete certificate
sudo certbot delete --cert-name yourdomain.com
```

### Intrusion Prevention

#### Fail2ban Setup

```bash
# Install fail2ban
sudo apt install fail2ban

# Copy default configuration
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local

# Edit configuration
sudo nano /etc/fail2ban/jail.local

# Enable and start
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Check status
sudo fail2ban-client status

# Unban IP address
sudo fail2ban-client set nginx-http-auth unbanip 192.168.1.100
```

#### Fail2ban Configuration for ExpertSystem

```bash
# Create custom jail
sudo nano /etc/fail2ban/jail.d/expertsystem.conf
```

```ini
[expertsystem]
enabled = true
port = 80,443,8501
filter = expertsystem
logpath = /opt/expert-system/logs/app.log
maxretry = 5
bantime = 3600
```

### Access Control

#### IP Whitelisting (Nginx)

```bash
# Edit nginx configuration
sudo nano /etc/nginx/sites-available/expert-system

# Add to server block:
allow 192.168.1.0/24;  # Allow local network
allow 10.0.0.100;       # Allow specific IP
deny all;                # Deny everyone else

# Reload nginx
sudo systemctl reload nginx
```

#### Basic Authentication

```bash
# Install apache2-utils
sudo apt install apache2-utils

# Create password file
sudo htpasswd -c /etc/nginx/.htpasswd admin

# Add more users
sudo htpasswd /etc/nginx/.htpasswd user2

# Edit nginx config
sudo nano /etc/nginx/sites-available/expert-system

# Add to location block:
auth_basic "ExpertSystem Access";
auth_basic_user_file /etc/nginx/.htpasswd;

# Reload nginx
sudo systemctl reload nginx
```

### System Security

#### Security Updates

```bash
# Install automatic updates
sudo apt install unattended-upgrades apt-listchanges

# Configure automatic updates
sudo dpkg-reconfigure -plow unattended-upgrades

# Manual updates
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
```

#### File Permissions

```bash
# Secure installation directory
sudo chown -R expert-system:expert-system /opt/expert-system
sudo chmod -R 750 /opt/expert-system

# Secure configuration files
sudo chmod 640 /opt/expert-system/config.py
sudo chmod 600 /opt/expert-system/.env

# Secure data directory
sudo chmod 750 /opt/expert-system/data
sudo chmod -R 640 /opt/expert-system/data/domains/*/catalog.json
```

#### Security Audit

```bash
# Check for security issues
sudo apt install lynis
sudo lynis audit system

# Check open ports
sudo netstat -tulnp
sudo ss -tulnp

# Check failed login attempts
sudo lastb

# Check successful logins
last
```

---

## ⚡ Performance Tuning

### Application Optimization

#### Cache Configuration

```bash
# Edit application configuration
sudo nano /opt/expert-system/config.py

# Adjust cache settings:
QUERY_CACHE_SIZE = 100      # Increase for more query caching
EMBEDDING_CACHE_SIZE = 1000  # Increase for more embedding caching
CACHE_TTL = 3600            # Cache time-to-live in seconds
```

#### Resource Limits

```bash
# Edit systemd service
sudo nano /etc/systemd/system/expert-system.service

# Add resource limits:
[Service]
MemoryLimit=8G
CPUQuota=400%

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart expert-system
```

### Database Optimization

#### ChromaDB Performance

```bash
# Optimize vector database
# Run periodically to maintain performance

sudo -u expert-system python3 << 'EOF'
import chromadb
client = chromadb.PersistentClient(path="/opt/expert-system/data/domains/*/vector_db")
# Database optimization happens automatically
EOF
```

#### Disk I/O Optimization

```bash
# Check disk performance
sudo iostat -x 1 5

# Mount options for better performance (add to /etc/fstab)
/dev/sda1  /opt/expert-system  ext4  noatime,nodiratime  0  2

# Remount with new options
sudo mount -o remount /opt/expert-system
```

### System Performance

#### Memory Optimization

```bash
# Check memory usage
free -h
vmstat 1 5

# Adjust swappiness (lower = use less swap)
sudo sysctl vm.swappiness=10
# Make permanent:
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
```

#### CPU Optimization

```bash
# Check CPU usage
mpstat 1 5

# Set CPU governor to performance
sudo cpupower frequency-set -g performance

# Or use powersave for efficiency
sudo cpupower frequency-set -g powersave
```

### Network Optimization

#### Nginx Performance Tuning

```bash
# Edit nginx configuration
sudo nano /etc/nginx/nginx.conf

# Optimize worker processes and connections:
worker_processes auto;
worker_connections 2048;
keepalive_timeout 65;
client_max_body_size 100M;

# Enable gzip compression:
gzip on;
gzip_vary on;
gzip_types text/plain text/css application/json application/javascript;

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

---

## 🔄 Updates and Maintenance

### Application Updates

#### Standard Installation

```bash
# Run update script
sudo /opt/expert-system/scripts/update.sh

# Or manual update:
cd /opt/expert-system
sudo git pull origin main
sudo -u expert-system pip install --upgrade -r requirements.txt
sudo systemctl restart expert-system
```

#### Docker Deployment

```bash
# Pull latest images
docker-compose pull

# Rebuild and restart
docker-compose up -d --build

# Or use update script
./scripts/docker-update.sh
```

### System Updates

```bash
# Update package lists
sudo apt update

# Upgrade packages
sudo apt upgrade -y

# Clean up
sudo apt autoremove -y
sudo apt autoclean

# Restart if kernel updated
sudo reboot
```

### Dependency Updates

```bash
# Update Python packages
cd /opt/expert-system
sudo -u expert-system pip list --outdated
sudo -u expert-system pip install --upgrade <package-name>

# Update all packages (careful!)
sudo -u expert-system pip install --upgrade -r requirements.txt
```

### Database Maintenance

```bash
# Vacuum and optimize
sudo -u expert-system sqlite3 /opt/expert-system/data/domains/*/vector_db/chroma.sqlite3 "VACUUM;"

# Check database integrity
sudo -u expert-system sqlite3 /opt/expert-system/data/domains/*/vector_db/chroma.sqlite3 "PRAGMA integrity_check;"
```

### Log Rotation

```bash
# Configure logrotate
sudo nano /etc/logrotate.d/expert-system
```

```
/opt/expert-system/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 expert-system expert-system
    sharedscripts
    postrotate
        systemctl reload expert-system > /dev/null 2>&1 || true
    endscript
}
```

```bash
# Test logrotate configuration
sudo logrotate -d /etc/logrotate.d/expert-system

# Force rotation
sudo logrotate -f /etc/logrotate.d/expert-system
```

### Cleanup Tasks

```bash
# Clean old backups (keep last 7 days)
find /opt/expert-system/backups/ -name "*.tar.gz" -mtime +7 -delete

# Clean cache
rm -rf /opt/expert-system/data/cache/*

# Clean temporary files
rm -rf /opt/expert-system/tmp/*

# Clean Docker (if used)
docker system prune -a -f
docker volume prune -f
```

---

## 🆘 Troubleshooting

### Service Won't Start

#### Check Service Status

```bash
# View service status
sudo systemctl status expert-system

# Check for errors
sudo journalctl -u expert-system -n 50 --no-pager

# Verify configuration
cd /opt/expert-system
python3 -c "import config; print('Config OK')"
```

#### Common Issues

**Port Already in Use:**
```bash
# Find what's using the port
sudo lsof -i :8501
sudo netstat -tulnp | grep 8501

# Kill process if needed
sudo kill <PID>

# Or change port in configuration
sudo nano /opt/expert-system/.streamlit/config.toml
```

**Permission Errors:**
```bash
# Fix ownership
sudo chown -R expert-system:expert-system /opt/expert-system

# Fix permissions
sudo chmod -R 750 /opt/expert-system
```

**Missing Dependencies:**
```bash
# Reinstall dependencies
cd /opt/expert-system
sudo -u expert-system pip install -r requirements.txt
```

### Application Errors

#### Ollama Connection Failed

```bash
# Check Ollama status
sudo systemctl status ollama

# Test Ollama connection
curl http://localhost:11434/api/version

# Check available models
ollama list

# Pull model if missing
ollama pull llama2

# Restart Ollama
sudo systemctl restart ollama
```

#### Database Errors

```bash
# Check database permissions
ls -la /opt/expert-system/data/domains/*/vector_db/

# Fix permissions
sudo chown -R expert-system:expert-system /opt/expert-system/data

# Rebuild database if corrupted
rm -rf /opt/expert-system/data/domains/*/vector_db/*
# Then re-upload documents in application
```

#### Memory Issues

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Kill heavy processes if needed
sudo systemctl restart expert-system

# Increase swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Performance Issues

#### Slow Queries

```bash
# Check system load
uptime
top

# Monitor application
sudo journalctl -u expert-system -f

# Clear cache
rm -rf /opt/expert-system/data/cache/*
sudo systemctl restart expert-system
```

#### High CPU Usage

```bash
# Identify process
top
htop

# Check for runaway processes
ps aux | grep python

# Restart application
sudo systemctl restart expert-system
```

#### Disk Space Issues

```bash
# Check disk usage
df -h

# Find large directories
du -sh /opt/expert-system/* | sort -h

# Clean up:
# - Old backups
find /opt/expert-system/backups/ -mtime +7 -delete
# - Logs
sudo truncate -s 0 /opt/expert-system/logs/*.log
# - Cache
rm -rf /opt/expert-system/data/cache/*
```

### Network Issues

#### Cannot Access Web Interface

```bash
# Check nginx status
sudo systemctl status nginx

# Test nginx configuration
sudo nginx -t

# Check firewall
sudo ufw status

# Check if port is listening
sudo netstat -tulnp | grep :80
sudo netstat -tulnp | grep :443

# Test locally
curl http://localhost
```

#### SSL Certificate Issues

```bash
# Check certificate status
sudo certbot certificates

# Renew certificate
sudo certbot renew

# Check nginx SSL configuration
sudo nano /etc/nginx/sites-available/expert-system

# Test SSL
openssl s_client -connect yourdomain.com:443
```

### Docker Issues

#### Containers Won't Start

```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs expert-system-app

# Restart containers
docker-compose restart

# Rebuild if needed
docker-compose down
docker-compose up -d --build
```

#### Volume Issues

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect expert-system_data

# Remove and recreate volume (WARNING: data loss)
docker-compose down -v
docker-compose up -d
```

### Diagnostic Commands

```bash
# System information
./setup.sh info

# Service status
./setup.sh status

# Update installation
./setup.sh update

# View all logs
sudo journalctl -xe

# Check all services
sudo systemctl --type=service --state=running | grep -E 'expert|ollama|nginx'
```

---

## 📞 Getting Administrative Help

### Log Collection

Before seeking support, collect relevant logs:

```bash
# Create support bundle
tar -czf support-bundle-$(date +%Y%m%d).tar.gz \
  /opt/expert-system/logs/ \
  /var/log/nginx/ \
  /etc/systemd/system/expert-system.service

# Share support bundle with support team
```

### System Information

```bash
# Gather system info
./setup.sh info > system-info.txt

# Include in support request
cat system-info.txt
```

### Documentation References

- **Installation**: See [INSTALLATION.md](INSTALLATION.md)
- **User Guide**: See [USERGUIDE.md](USERGUIDE.md)
- **Development**: See [DEVELOPER.md](DEVELOPER.md)
- **Main Documentation**: See [README.md](../README.md)

---

## 📝 Maintenance Checklist

### Daily

- [ ] Check service status
- [ ] Review error logs
- [ ] Monitor disk space

### Weekly

- [ ] Review backup success
- [ ] Check system updates
- [ ] Monitor performance metrics
- [ ] Review security logs

### Monthly

- [ ] Test backup restoration
- [ ] Update system packages
- [ ] Review user access logs
- [ ] Clean old backups and logs
- [ ] Update SSL certificates (if needed)
- [ ] Performance optimization review

### Quarterly

- [ ] Full system security audit
- [ ] Review and update documentation
- [ ] Capacity planning review
- [ ] Disaster recovery test

---

**For additional help, consult the main documentation or review system logs for detailed error information.**
