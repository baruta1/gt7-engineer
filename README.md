# GT7 Race Engineer

A Discord-based real-time racing engineer for Gran Turismo 7. Features an AI-powered engineer named **Aoife** who provides F1-style radio communications, automatic telemetry updates, and responds to voice commands during races.

![GT7 Engineer](Image.png)

## Features

- **F1-Style Radio Communications**: Authentic radio sound effects with engineer voice processing
- **Live Telemetry Alerts**: Automatic updates for lap times, fuel levels, tire wear, and position changes
- **Voice Commands**: Say "Radio" followed by your question to get real-time stats
- **Smart Lap Analysis**: Sector comparisons, driving style feedback, and performance insights
- **Incident Detection**: Automatic alerts for spins, crashes, and off-track incidents
- **Piper TTS**: Fast, high-quality British voice synthesis using local Piper models

## Requirements

- Python 3.10+
- PlayStation 5 with Gran Turismo 7
- Discord bot with voice permissions
- OpenAI API key
- PS5 and computer on the same network

## Installation

### Linux/WSL

```bash
git clone https://github.com/baruta1/gt7-engineer.git
cd gt7-engineer
chmod +x setup.sh
./setup.sh
```

### Windows

```cmd
git clone https://github.com/baruta1/gt7-engineer.git
cd gt7-engineer
setup_windows.bat
```

### Manual Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download Piper TTS model:
   ```bash
   mkdir -p piper_models
   # Download en_GB-alba-medium from:
   # https://github.com/rhasspy/piper/releases
   # Place .onnx and .onnx.json files in piper_models/
   ```

4. Create `.env` file with your credentials:
   ```ini
   DISCORD_TOKEN=your_discord_bot_token
   OPENAI_API_KEY=your_openai_api_key
   DISCORD_CHANNEL_ID=your_voice_channel_id
   ```

## Configuration

Edit `telemetry_server.py` to set your PS5 IP address:
```python
def __init__(self, ps5_ip="192.168.1.XXX"):
```

## Usage

### Starting the Engineer

**Linux/WSL:**
```bash
./start_engineer.sh
```

**Windows:**
```cmd
GT7_Engineer.bat
```

**Or manually:**
```bash
python launcher.py
```

### During Racing

1. Start GT7 on your PS5
2. Run the engineer - it will auto-connect to telemetry
3. The bot automatically joins your Discord voice channel
4. **Automatic updates**: Lap times, fuel alerts, position changes
5. **Voice commands**: Say "Radio, what's my fuel?" or "Radio, tire status?"

### Voice Commands

- "Radio, fuel status" - Current fuel level and laps remaining
- "Radio, tire wear" - Tire condition report
- "Radio, what position am I?" - Current race position
- "Radio, how am I doing?" - Overall performance summary
- "Radio, gap to leader?" - Time gap information

## WSL Network Setup

If running on WSL2, enable mirrored networking for PS5 access:

1. Create/edit `C:\Users\<YourName>\.wslconfig`:
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```

2. Restart WSL: `wsl --shutdown` then reopen

3. Add Windows Firewall rule for UDP port 33740

## Repository Structure

```
gt7-engineer/
├── GT7_Radio_GenAI.py     # Main bot logic and AI engineer
├── launcher.py            # Telemetry connector and startup
├── telemetry_server.py    # PS5 telemetry listener
├── gui_launcher.py        # Optional GUI launcher
├── requirements.txt       # Python dependencies
├── Radio/                 # Sound effects (Start.flac, End1.wav)
├── Message_Confirmation.wav
├── setup.sh              # Linux/WSL setup script
├── setup_windows.bat     # Windows setup script
├── start_engineer.sh     # Linux/WSL run script
└── GT7_Engineer.bat      # Windows run script
```

## Dependencies

- discord.py & discord-ext-voice-recv
- openai
- faster-whisper
- piper-tts
- pydub
- gt-telem
- python-dotenv

## Troubleshooting

**Telemetry not connecting:**
- Ensure PS5 and computer are on the same network
- Check PS5 IP address in `telemetry_server.py`
- For WSL, enable mirrored networking mode

**Voice not working:**
- Check Discord bot has voice permissions
- Verify channel ID in `.env`
- Ensure ffmpeg is installed

**TTS errors:**
- Download Piper model files to `piper_models/`
- Check model path in `GT7_Radio_GenAI.py`

## License

MIT
