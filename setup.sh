#!/bin/bash
# GT7 Race Engineer - First-time Setup Script
# Run this once to install all dependencies

cd "$(dirname "$0")"

echo "=== GT7 Race Engineer Setup ==="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed!"
    echo "Please install Python 3.9+ first."
    exit 1
fi

# Check for FFmpeg (required for audio)
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Installing..."
    sudo apt-get update && sudo apt-get install -y ffmpeg
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ""
    echo "=== IMPORTANT: Configure your .env file ==="
    echo "Edit the .env file with your credentials:"
    echo "  - DISCORD_TOKEN: Your Discord bot token"
    echo "  - OPENAI_API_KEY: Your OpenAI API key"
    echo "  - AUTO_CHANNEL_ID: Discord voice channel ID"
    echo "  - DRIVER_USER_ID: Your Discord user ID"
    echo ""
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your credentials"
echo "2. Run ./start_engineer.sh to start the bot"
echo "   Or double-click GT7_Engineer.bat from Windows"
