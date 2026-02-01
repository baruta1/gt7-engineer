# launcher.py
import os
import sys
import time
from pathlib import Path

# Import telemetry BEFORE GT7_Radio_GenAI (which applies nest_asyncio)
from telemetry_server import TelemetryServer

os.environ.setdefault("PYTHONUNBUFFERED", "1")

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import logging
logging.basicConfig(filename="engineer_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

MAX_ATTEMPTS = 3
DELAY_BETWEEN = 5  # seconds


def set_working_directory():
    # Ensure the working directory is the script's location
    os.chdir(Path(__file__).resolve().parent)


def connect_telemetry():
    for attempt in range(1, MAX_ATTEMPTS + 1):
        tel = TelemetryServer()
        try:
            tel.start()
            # Wait up to 5 seconds for telemetry data
            for _ in range(10):
                time.sleep(0.5)
                if tel.get_latest() is not None:
                    print(f"✅ Telemetry connected on attempt {attempt}", flush=True)
                    return tel
            print(f"⚠️  Telemetry attempt {attempt}: no packets received", flush=True)
            tel.stop()
            time.sleep(DELAY_BETWEEN)
        except Exception as e:
            print(f"⚠️  Telemetry attempt {attempt} failed: {e}", flush=True)
            try:
                tel.stop()
            except:
                pass
            time.sleep(DELAY_BETWEEN)
    return None


def show_error_message():
    print("=" * 50)
    print("❌ ERROR: Couldn't find your PS5 telemetry after 3 tries.")
    print("Please check:")
    print("  - Your PS5 is on and GT7 is running")
    print("  - PS5 and computer are on the same network")
    print("  - Try restarting your PS5")
    print("=" * 50)


def main():
    set_working_directory()
    tel = connect_telemetry()
    if not tel:
        show_error_message()
        sys.exit(1)

    # Import AFTER telemetry is connected (nest_asyncio applied in this module)
    import GT7_Radio_GenAI
    GT7_Radio_GenAI.run_with_telemetry(tel)


if __name__ == "__main__":
    main()