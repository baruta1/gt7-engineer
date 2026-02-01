@echo off
REM GT7 Race Engineer - Windows Setup Script
REM Run this ONCE to install dependencies on Windows

echo === GT7 Race Engineer Windows Setup ===
echo.

REM Check Python
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv_win

REM Activate and install
echo Installing dependencies...
call venv_win\Scripts\activate.bat
python -m pip install --upgrade pip
pip install discord.py[voice] discord-ext-voice-recv faster-whisper openai pydub edge-tts python-dotenv pillow gt-telem nest-asyncio aiohttp async-timeout inflect

echo.
echo === Setup Complete! ===
echo.
echo Now double-click GT7_Engineer_Windows.bat to run the engineer.
pause
