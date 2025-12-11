#!/bin/bash

# VANET RL Exhibition Dashboard - Startup Script
# ==============================================

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     VANET RL Exhibition Dashboard - Setup & Start         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    cd "$PROJECT_ROOT"
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$PROJECT_ROOT/venv/bin/activate"

# Install dashboard dependencies
echo ""
echo "Installing dashboard dependencies..."
cd "$SCRIPT_DIR"
pip install -q -r requirements.txt

# Check if agent dependencies are installed
echo ""
echo "Checking agent dependencies..."
cd "$PROJECT_ROOT/agent"
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
fi

# Check for trained models
echo ""
echo "Checking for trained models..."
if [ ! -f "$PROJECT_ROOT/agent/models/ppo_dual_final_20251116_194131.pth" ]; then
    echo "⚠️  Warning: PPO model not found!"
    echo "   Expected: agent/models/ppo_dual_final_20251116_194131.pth"
fi

if [ ! -f "$PROJECT_ROOT/agent/models/dqn_dual_final_20251116_192041.pth" ]; then
    echo "⚠️  Warning: DQN model not found!"
    echo "   Expected: agent/models/dqn_dual_final_20251116_192041.pth"
fi

# Check if NS3 is built
echo ""
echo "Checking NS3 simulation..."
NS3_DIR="/home/tharindu/tarballs/ns-allinone-3.44/ns-3.44"
if [ ! -d "$NS3_DIR/build" ]; then
    echo "⚠️  Warning: NS3 build directory not found at $NS3_DIR/build"
    echo "   Make sure NS3 is built and ready"
    echo "   Run: cd $NS3_DIR && ./ns3 build"
else
    echo "✅ NS3 found at: $NS3_DIR"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Setup complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Starting dashboard server..."
echo ""
echo "Dashboard will be available at:"
echo "  🌐 http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Start the server
cd "$SCRIPT_DIR"
python server.py

# Deactivate virtual environment on exit
deactivate
