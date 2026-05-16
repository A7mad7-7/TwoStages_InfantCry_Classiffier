#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# 🍼 Smart Crib — Raspberry Pi Setup & Auto-Start Script (Bookworm / NetworkManager)
# ═══════════════════════════════════════════════════════════════════════

set -e # Exit immediately if a command exits with a non-zero status

echo "========================================================="
echo "  Setting up Smart Crib Edge AI + Wi-Fi Hotspot on Pi    "
echo "  Uses NetworkManager (nmcli) for Raspbian >= Bookworm   "
echo "========================================================="

# 1. System Dependencies (Audio + Networking)
echo ""
echo "[1/4] Installing system dependencies (portaudio, network-manager)..."
sudo apt-get update
sudo apt-get install -y portaudio19-dev libsndfile1-dev python3-pip python3-venv network-manager

# 2. Python Virtual Environment
echo ""
echo "[2/4] Setting up isolated Python virtual environment..."
python3 -m venv venv_pi
source venv_pi/bin/activate

# 3. Install Python requirements
echo ""
echo "[3/4] Installing lightweight Python libraries (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements_pi.txt

# 4. Configure Wi-Fi Access Point (NetworkManager via nmcli)
echo ""
echo "[4/4] Configuring Raspberry Pi as a Wi-Fi Access Point (SmartCrib_WiFi)..."

# Ensure Wi-Fi isn't blocked by the OS
sudo rfkill unblock wlan || true

# Stop NetworkManager from automatically connecting to any known home WiFis
sudo nmcli radio wifi on

# Delete any previous SmartCrib connection profile if it exists
sudo nmcli connection delete SmartCrib_Hotspot 2>/dev/null || true

echo "     → Creating Open Hotspot ('SmartCrib_WiFi') on 192.168.4.1..."

# Create a master connection profile
sudo nmcli connection add type wifi ifname wlan0 con-name SmartCrib_Hotspot autoconnect yes ssid SmartCrib_WiFi
# Set it to act as an Access Point (AP) in the standard 2.4GHz bands
sudo nmcli connection modify SmartCrib_Hotspot 802-11-wireless.mode ap 802-11-wireless.band bg
# Assign it a static IP (internal gateway) of 192.168.4.1
sudo nmcli connection modify SmartCrib_Hotspot ipv4.method shared ipv4.addresses 192.168.4.1/24
# Make it an OPEN network (no password required)
sudo nmcli connection modify SmartCrib_Hotspot wifi-sec.key-mgmt none

# Turn on the Hotspot right now
sudo nmcli connection up SmartCrib_Hotspot || echo "[WARN] Hotspot start pending reboot."

# 5. Create Systemd Service for Auto-start on Boot
echo ""
echo "[5/5] Creating systemd service for auto-boot (smartcrib.service)..."

SERVICE_FILE="/etc/systemd/system/smartcrib.service"
CURRENT_DIR=$(pwd)
USER_NAME=$(whoami)

sudo bash -c "cat > $SERVICE_FILE" <<EOL
[Unit]
Description=Smart Crib - Edge AI Brain
# Start after networking is fully up
After=network-online.target sound.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$CURRENT_DIR
Environment="PATH=$CURRENT_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$CURRENT_DIR/venv/bin/python $CURRENT_DIR/main_system.py

Restart=always
RestartSec=10

StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=smartcrib

[Install]
WantedBy=multi-user.target
EOL

sudo systemctl daemon-reload
sudo systemctl enable smartcrib.service

echo ""
echo "========================================================="
echo "  ✅ Setup Complete!                                     "
echo "========================================================="
echo "  1. Reboot the Pi:             sudo reboot"
echo "  2. Connect Parent Phone to:   SmartCrib_WiFi (Open Network)"
echo "  3. Open Browser to:           ch"
echo "========================================================="
