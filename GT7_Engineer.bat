@echo off
REM GT7 Race Engineer - Windows Launcher for WSL
REM Double-click this file to start the engineer!

cd /d %USERPROFILE%
echo Starting GT7 Race Engineer...
echo.
wsl -e bash -c "cd /home/paul/gt7-engineer && source venv/bin/activate && python launcher.py"
pause
