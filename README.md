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
- Discord account and a Discord server you control
- OpenAI API key
- PS5 and computer on the same network

---

## Step 1: Create a Discord Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application** and give it a name (e.g., "GT7 Engineer")
3. Go to the **Bot** tab on the left sidebar
4. Click **Add Bot**, then confirm
5. Under the bot's username, click **Reset Token** and copy your bot token (save this for later!)
6. Scroll down and enable these **Privileged Gateway Intents**:
   - Presence Intent
   - Server Members Intent
   - Message Content Intent
7. Go to **OAuth2 > URL Generator**:
   - Under **Scopes**, check `bot`
   - Under **Bot Permissions**, check: `Connect`, `Speak`, `Send Messages`, `Read Message History`
8. Copy the generated URL at the bottom and open it in your browser to invite the bot to your server

---

## Step 2: Find Your Discord IDs

You'll need to enable **Developer Mode** in Discord to copy IDs:

1. Open Discord (desktop app or browser)
2. Go to **User Settings** (gear icon near your username)
3. Scroll down to **Advanced** under App Settings
4. Toggle **Developer Mode** ON

Now you can right-click on things to copy their IDs:

### Voice Channel ID (optional - for auto-join)
1. Right-click on the voice channel where you want the bot to join
2. Click **Copy Channel ID**
3. Save this number (e.g., `1234567890123456789`)

### Your User ID (optional - for auto-join)
1. Right-click on your own username in the member list
2. Click **Copy User ID**
3. Save this number

---

## Step 3: Get an OpenAI API Key

1. Go to [OpenAI's API platform](https://platform.openai.com/)
2. Sign up or log in
3. Go to **API Keys** and create a new key
4. Copy and save the key (you won't see it again!)

---

## Step 4: Find Your PS5 IP Address

1. On your PS5, go to **Settings > Network > View Connection Status**
2. Note your **IPv4 Address** (e.g., `192.168.1.108`)

---

## Step 5: Installation

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
   # Download en_GB-alba-medium.onnx and en_GB-alba-medium.onnx.json from:
   # https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_GB/alba/medium
   # Place both files in the piper_models/ folder
   ```

---

## Step 6: Configuration

### Create your `.env` file

Create a file called `.env` in the project folder with your credentials:

```ini
# Required
DISCORD_TOKEN=your_discord_bot_token_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional - for auto-joining voice channel on startup
AUTO_CHANNEL_ID=your_voice_channel_id
DRIVER_USER_ID=your_discord_user_id
```

**Note:** The `AUTO_CHANNEL_ID` and `DRIVER_USER_ID` are optional. If you don't set them, you'll need to manually type `!engineer` in Discord to make the bot join your voice channel.

### Set your PS5 IP address

Edit `telemetry_server.py` and change the IP to match your PS5:

```python
def __init__(self, ps5_ip="192.168.1.108"):  # Change this to your PS5 IP
```

---

## Step 7: Running the Engineer

### Start the bot

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

### Connect to your race

1. **Start GT7** on your PS5
2. **Run the engineer** - it will connect to your PS5 telemetry automatically
3. **Join a Discord voice channel** on your computer or phone
4. If you didn't set up auto-join, type `!engineer` in any text channel to make the bot join your voice channel
5. **Transfer Discord audio to your PS5** (see below)

### Getting Discord Audio on Your PS5

The engineer speaks through Discord, so you need to hear Discord while racing. Options:

**Option A: Discord on your phone (easiest)**
- Join the same voice channel on your phone
- Use earbuds or put your phone near you while racing

**Option B: Discord on PC with PS5 Remote Play audio**
- Use PS5 Remote Play on your PC
- Route game audio through your PC speakers/headset
- You'll hear both Discord and the game

**Option C: Audio mixer**
- Use a headset mixer to combine PS5 audio and PC/phone audio

---

## During Racing

### Automatic Updates

The engineer will automatically announce:
- Lap times and deltas
- Fuel warnings (50%, 20%, 10% remaining)
- Position changes
- Incidents (spins, crashes)

### Voice Commands

Say "Radio" followed by your question:

- **"Radio, fuel status"** - Current fuel level and laps remaining
- **"Radio, tire wear"** - Tire condition report
- **"Radio, what position am I?"** - Current race position
- **"Radio, how am I doing?"** - Overall performance summary
- **"Radio, gap to leader?"** - Time gap information

---

## WSL Network Setup

If running on WSL2, you need to enable mirrored networking so WSL can see your PS5:

1. Create or edit `C:\Users\<YourName>\.wslconfig`:
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```

2. Restart WSL:
   ```cmd
   wsl --shutdown
   ```
   Then reopen your WSL terminal.

3. Add a Windows Firewall rule to allow UDP port 33740 (GT7 telemetry):
   - Open Windows Defender Firewall
   - Click "Advanced settings"
   - Right-click "Inbound Rules" > "New Rule"
   - Select "Port" > "UDP" > Specific port: `33740`
   - Allow the connection > Apply to all profiles > Name it "GT7 Telemetry"

---

## Repository Structure

```
gt7-engineer/
├── GT7_Radio_GenAI.py     # Main bot logic and AI engineer
├── launcher.py            # Telemetry connector and startup
├── telemetry_server.py    # PS5 telemetry listener
├── gui_launcher.py        # Optional GUI launcher
├── requirements.txt       # Python dependencies
├── Radio/                 # Sound effects (Start.flac, End1.wav)
├── piper_models/          # TTS voice models (download separately)
├── .env                   # Your credentials (create this)
├── setup.sh              # Linux/WSL setup script
├── setup_windows.bat     # Windows setup script
├── start_engineer.sh     # Linux/WSL run script
└── GT7_Engineer.bat      # Windows run script
```

---

## Troubleshooting

**Bot won't connect to Discord:**
- Double-check your `DISCORD_TOKEN` in `.env`
- Make sure you enabled the required Privileged Gateway Intents
- Verify the bot was invited to your server with the correct permissions

**Telemetry not connecting:**
- Ensure PS5 and computer are on the same network
- Verify the PS5 IP address in `telemetry_server.py`
- Make sure GT7 is running (not just at the PS5 home screen)
- For WSL, enable mirrored networking mode and add the firewall rule

**Bot won't join voice channel:**
- Make sure you're in a voice channel first
- Try typing `!engineer` in a text channel
- Check that the bot has Connect and Speak permissions

**Can't hear the engineer:**
- Make sure you're in the same voice channel as the bot
- Check your Discord audio settings
- Ensure ffmpeg is installed (`sudo apt install ffmpeg` on Linux)

**TTS errors:**
- Download both `.onnx` and `.onnx.json` Piper model files
- Place them in the `piper_models/` folder
- Check the model path in `GT7_Radio_GenAI.py`

---

## Dependencies

- discord.py & discord-ext-voice-recv
- openai
- faster-whisper
- piper-tts
- pydub
- gt-telem
- python-dotenv

---

## License

MIT
