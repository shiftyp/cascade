# CASCADE Modem - Deployment Guide

Production deployment instructions for Linux, Windows, and macOS.

---

## Production Checklist

Before deploying:

- [ ] Backend configuration (`.env`) set correctly
- [ ] Hamlib radio connection tested
- [ ] Audio devices configured
- [ ] Frontend built (`npm run build`)
- [ ] Firewall rules configured
- [ ] Backup strategy in place

---

## Linux Deployment (systemd)

Recommended for Raspberry Pi and Linux servers.

### 1. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3-pip python3-venv
sudo apt install -y libhamlib4 libhamlib-dev  # Hamlib
sudo apt install -y portaudio19-dev           # Audio
sudo apt install -y nodejs npm                 # Frontend

# Fedora/RHEL
sudo dnf install -y python3.11 python3-pip
sudo dnf install -y hamlib hamlib-devel
sudo dnf install -y portaudio-devel
sudo dnf install -y nodejs npm
```

### 2. Create Service User

```bash
# Create dedicated user (no login shell)
sudo useradd -r -s /bin/false cascade

# Create directories
sudo mkdir -p /opt/cascade
sudo chown cascade:cascade /opt/cascade
```

### 3. Install CASCADE

```bash
# Clone and setup (as cascade user)
sudo -u cascade git clone https://github.com/yourorg/cascade.git /opt/cascade
cd /opt/cascade/modules/modem

# Backend
cd backend
sudo -u cascade python3 -m venv venv
sudo -u cascade ./venv/bin/pip install -r requirements.txt

# Frontend
cd ../frontend
sudo -u cascade npm ci --production
sudo -u cascade npm run build
```

### 4. Configuration

Create `/opt/cascade/modules/modem/backend/.env`:

```env
CALLSIGN=W1ABC
GRID_SQUARE=FN42mc
HARDWARE_TIER=rpi4

RADIO_MODEL=3037
RADIO_PORT=/dev/ttyUSB0
RADIO_BAUD=9600

FREQUENCY=14074000
MODE=USB

AUDIO_INPUT_DEVICE=1
AUDIO_OUTPUT_DEVICE=1

HOST=0.0.0.0
PORT=8000
DEBUG=false
```

Set ownership:
```bash
sudo chown cascade:cascade /opt/cascade/modules/modem/backend/.env
sudo chmod 600 /opt/cascade/modules/modem/backend/.env
```

### 5. systemd Service

Create `/etc/systemd/system/cascade-modem.service`:

```ini
[Unit]
Description=CASCADE Modem Server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=cascade
Group=cascade
WorkingDirectory=/opt/cascade/modules/modem/backend
Environment=PATH=/opt/cascade/modules/modem/backend/venv/bin:/usr/bin
ExecStart=/opt/cascade/modules/modem/backend/venv/bin/python main.py
Restart=always
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/cascade

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cascade-modem

[Install]
WantedBy=multi-user.target
```

### 6. Start Service

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable cascade-modem
sudo systemctl start cascade-modem

# Check status
sudo systemctl status cascade-modem

# View logs
sudo journalctl -u cascade-modem -f
```

### 7. Access

Open browser to:
- Local: `http://localhost:8000`
- LAN: `http://YOUR_IP:8000`

### 8. Firewall (ufw)

```bash
# Allow CASCADE port
sudo ufw allow 8000/tcp

# If remote access needed
sudo ufw allow from 192.168.1.0/24 to any port 8000
```

---

## Windows Deployment (NSSM)

### 1. Install Dependencies

- **Python 3.11+:** https://www.python.org/downloads/
- **Hamlib for Windows:** https://github.com/Hamlib/Hamlib/releases
  - Download `hamlib-w64-4.x.zip`
  - Extract to `C:\Program Files\Hamlib`
  - Add to PATH: `C:\Program Files\Hamlib\bin`
- **Node.js:** https://nodejs.org/

### 2. Install CASCADE

```powershell
# Clone repository
git clone https://github.com/yourorg/cascade.git C:\CASCADE
cd C:\CASCADE\modules\modem

# Backend
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd ..\frontend
npm ci --production
npm run build
```

### 3. Configuration

Create `C:\CASCADE\modules\modem\backend\.env`:

```env
CALLSIGN=W1ABC
GRID_SQUARE=FN42mc

RADIO_MODEL=3037
RADIO_PORT=COM3
RADIO_BAUD=9600

FREQUENCY=14074000

HOST=0.0.0.0
PORT=8000
```

### 4. Install as Windows Service (NSSM)

```powershell
# Download NSSM: https://nssm.cc/download
# Extract nssm.exe to C:\Windows\System32

# Install service
nssm install CASCADE-Modem `
  "C:\CASCADE\modules\modem\backend\venv\Scripts\python.exe" `
  "C:\CASCADE\modules\modem\backend\main.py"

# Configure service
nssm set CASCADE-Modem AppDirectory "C:\CASCADE\modules\modem\backend"
nssm set CASCADE-Modem DisplayName "CASCADE Modem Server"
nssm set CASCADE-Modem Description "HF Digital Modem for CASCADE Protocol"
nssm set CASCADE-Modem Start SERVICE_AUTO_START

# Start service
nssm start CASCADE-Modem

# Check status
nssm status CASCADE-Modem
```

### 5. Windows Firewall

```powershell
# Allow inbound connections
New-NetFirewallRule -DisplayName "CASCADE Modem" `
  -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

### 6. Access

Open browser to `http://localhost:8000`

---

## macOS Deployment (launchd)

### 1. Install Dependencies

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11
brew install hamlib
brew install portaudio
brew install node
```

### 2. Install CASCADE

```bash
# Clone repository
git clone https://github.com/yourorg/cascade.git ~/CASCADE
cd ~/CASCADE/modules/modem

# Backend
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm ci --production
npm run build
```

### 3. Configuration

Create `~/CASCADE/modules/modem/backend/.env`:

```env
CALLSIGN=W1ABC
GRID_SQUARE=FN42mc

RADIO_MODEL=3037
RADIO_PORT=/dev/cu.usbserial-1234
RADIO_BAUD=9600

FREQUENCY=14074000

HOST=0.0.0.0
PORT=8000
```

### 4. launchd Service

Create `~/Library/LaunchAgents/com.cascade.modem.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.cascade.modem</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/CASCADE/modules/modem/backend/venv/bin/python</string>
        <string>/Users/YOUR_USERNAME/CASCADE/modules/modem/backend/main.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/YOUR_USERNAME/CASCADE/modules/modem/backend</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/Users/YOUR_USERNAME/CASCADE/cascade-modem.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USERNAME/CASCADE/cascade-modem-error.log</string>
</dict>
</plist>
```

Load service:
```bash
launchctl load ~/Library/LaunchAgents/com.cascade.modem.plist
launchctl start com.cascade.modem

# Check status
launchctl list | grep cascade
```

---

## Docker Deployment (Optional)

### Dockerfile

Create `modules/modem/Dockerfile`:

```dockerfile
# Multi-stage build

# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Backend
FROM python:3.11-slim
WORKDIR /app

# Install Hamlib and audio dependencies
RUN apt-get update && apt-get install -y \
    libhamlib4 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "main.py"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  cascade-modem:
    build: .
    container_name: cascade-modem
    restart: unless-stopped

    ports:
      - "8000:8000"

    devices:
      - /dev/ttyUSB0:/dev/ttyUSB0  # Serial port for radio
      - /dev/snd:/dev/snd          # Audio device

    environment:
      - CALLSIGN=W1ABC
      - GRID_SQUARE=FN42mc
      - RADIO_MODEL=3037
      - RADIO_PORT=/dev/ttyUSB0
      - FREQUENCY=14074000
      - HOST=0.0.0.0
      - PORT=8000

    volumes:
      - ./data:/app/data  # Persistent data

    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Run with Docker Compose

```bash
cd modules/modem
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Reverse Proxy (nginx)

For HTTPS access (recommended for internet-facing deployments):

### nginx Configuration

```nginx
server {
    listen 80;
    server_name cascade.example.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name cascade.example.com;

    # SSL certificates (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/cascade.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/cascade.example.com/privkey.pem;

    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Proxy to CASCADE backend
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://localhost:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

### Install SSL Certificate (Let's Encrypt)

```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d cascade.example.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

---

## Raspberry Pi Specific

### Performance Optimization

**1. GPU Memory (if headless):**

Edit `/boot/config.txt`:
```
gpu_mem=16  # Reduce GPU memory (headless)
```

**2. Swap File:**

```bash
# Increase swap (for 2GB RAM models)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

**3. Auto-start on Boot:**

Already handled by systemd service.

### GPIO PTT (Optional)

If using GPIO for PTT instead of Hamlib:

Edit backend code to use RPi.GPIO:

```python
# backend/hardware/ptt_control.py
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)  # GPIO 17 for PTT

def set_ptt(state: bool):
    GPIO.output(17, GPIO.HIGH if state else GPIO.LOW)
```

---

## Monitoring & Logging

### View Logs (Linux systemd)

```bash
# Real-time logs
sudo journalctl -u cascade-modem -f

# Last 100 lines
sudo journalctl -u cascade-modem -n 100

# Since boot
sudo journalctl -u cascade-modem -b

# Filter by time
sudo journalctl -u cascade-modem --since "1 hour ago"
```

### Log Rotation

Logs automatically rotated by systemd.

Configure in `/etc/systemd/journald.conf`:
```ini
[Journal]
SystemMaxUse=500M
MaxRetentionSec=1month
```

---

## Backup Strategy

### Files to Backup

- `/opt/cascade/modules/modem/backend/.env` - Configuration
- `/opt/cascade/data/` - Telemetry data (if storing locally)

### Automated Backup (cron)

```bash
# Create backup script
sudo nano /opt/cascade/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/backup/cascade"
DATE=$(date +%Y%m%d)

mkdir -p $BACKUP_DIR

# Backup configuration
cp /opt/cascade/modules/modem/backend/.env $BACKUP_DIR/env.$DATE

# Backup data
tar -czf $BACKUP_DIR/data.$DATE.tar.gz /opt/cascade/data

# Keep last 7 backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

Add to crontab:
```bash
sudo crontab -e
# Add:
0 2 * * * /opt/cascade/backup.sh
```

---

## Updates

### Update CASCADE

```bash
# Stop service
sudo systemctl stop cascade-modem

# Pull updates
cd /opt/cascade
sudo -u cascade git pull

# Update dependencies
cd modules/modem/backend
sudo -u cascade ./venv/bin/pip install -r requirements.txt

cd ../frontend
sudo -u cascade npm ci
sudo -u cascade npm run build

# Restart service
sudo systemctl start cascade-modem
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check status
sudo systemctl status cascade-modem

# Check logs
sudo journalctl -u cascade-modem -n 50

# Test manually
cd /opt/cascade/modules/modem/backend
sudo -u cascade ./venv/bin/python main.py
```

### Radio Not Connecting

```bash
# Check USB serial device
ls -l /dev/ttyUSB*

# Check permissions
sudo usermod -a -G dialout cascade

# Test with rigctl
rigctl -m 3037 -r /dev/ttyUSB0 -s 9600
```

### Audio Issues

```bash
# List audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Check ALSA (Linux)
aplay -l  # List playback devices
arecord -l  # List capture devices
```

---

## Security Considerations

### Firewall Rules

Only allow CASCADE from trusted networks:

```bash
# ufw (Ubuntu/Debian)
sudo ufw deny 8000
sudo ufw allow from 192.168.1.0/24 to any port 8000

# firewalld (Fedora/RHEL)
sudo firewall-cmd --add-rich-rule='rule family="ipv4" source address="192.168.1.0/24" port port="8000" protocol="tcp" accept' --permanent
sudo firewall-cmd --reload
```

### HTTPS (Required for Internet Access)

Use nginx reverse proxy with Let's Encrypt (see above).

**Never expose CASCADE directly to internet without HTTPS!**

---

## See Also

- **[README.md](README.md)** - Setup and usage guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design
- **[API.md](API.md)** - REST + WebSocket protocol
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guide

---

**Last updated:** 2025-10-04
