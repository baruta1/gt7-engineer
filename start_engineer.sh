#!/bin/bash
# GT7 Race Engineer - One-Click Launcher
# This script starts the GT7 Race Engineer with auto-connect to Discord

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with your credentials:"
    echo "  DISCORD_TOKEN=your_token"
    echo "  OPENAI_API_KEY=your_api_key"
    echo "  AUTO_CHANNEL_ID=your_channel_id"
    echo "  DRIVER_USER_ID=your_user_id"
    exit 1
fi

# Check for missing Message_Received.wav - use End1.wav as fallback
if [ ! -f "Message_Received.wav" ]; then
    echo "Creating Message_Received.wav symlink..."
    ln -sf Radio/End1.wav Message_Received.wav
fi

echo "Starting GT7 Race Engineer..."
echo "Auto-connect is enabled - bot will connect to your Discord channel automatically!"
echo ""

python launcher.py
