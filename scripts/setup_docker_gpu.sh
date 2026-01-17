#!/bin/bash
# setup_docker_gpu.sh - One-shot Docker + NVIDIA Container Toolkit installer
# For Ubuntu systems with NVIDIA GPUs (tested on Ubuntu 22.04)
#
# Usage:
#   sudo ./scripts/setup_docker_gpu.sh
#   # Then log out and back in for docker group permissions
#
# Verification:
#   docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

set -e

echo "=============================================="
echo "Docker + NVIDIA Container Toolkit Setup"
echo "=============================================="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

# Get the real user who invoked sudo
REAL_USER=${SUDO_USER:-$USER}

echo ""
echo "Step 1.1: Remove old Docker installations..."
apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

echo ""
echo "Step 1.2: Install prerequisites..."
apt-get update
apt-get install -y ca-certificates curl gnupg lsb-release

echo ""
echo "Step 1.3: Add Docker GPG key and repository..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

echo ""
echo "Step 1.4: Install Docker Engine..."
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo ""
echo "Step 1.5: Add user '$REAL_USER' to docker group..."
usermod -aG docker $REAL_USER

echo ""
echo "Step 1.6: Start and enable Docker service..."
systemctl start docker
systemctl enable docker

echo ""
echo "=============================================="
echo "Step 2: Install NVIDIA Container Toolkit"
echo "=============================================="

echo ""
echo "Step 2.1: Add NVIDIA repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo ""
echo "Step 2.2: Install NVIDIA Container Toolkit..."
apt-get update
apt-get install -y nvidia-container-toolkit

echo ""
echo "Step 2.3: Configure Docker runtime for NVIDIA..."
nvidia-ctk runtime configure --runtime=docker

echo ""
echo "Step 2.4: Restart Docker..."
systemctl restart docker

echo ""
echo "=============================================="
echo "Step 3: Verify Installation"
echo "=============================================="

echo ""
echo "Testing Docker..."
docker --version

echo ""
echo "Testing NVIDIA Container Toolkit..."
# Run as the real user to test group permissions might not work without re-login
# So we run directly with docker which is root at this point
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "=============================================="
echo "SUCCESS!"
echo "=============================================="
echo ""
echo "Docker and NVIDIA Container Toolkit are installed."
echo ""
echo "IMPORTANT: Please log out and log back in (or run 'newgrp docker')"
echo "           for the docker group permissions to take effect."
echo ""
echo "To verify after re-login, run:"
echo "  docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi"
echo ""
