import time
import threading
from gt_telem import TurismoClient
from gt_telem.errors.playstation_errors import PlayStationNotFoundError, PlayStationOnStandbyError

class TelemetryServer:
    def __init__(self, ps5_ip="192.168.1.108"):
        self.ps5_ip = ps5_ip
        self.tc = None
        self.latest = None
        self.running = False
        self.thread = None

    def _loop(self):
        while self.running:
            if self.tc.telemetry:
                self.latest = self.tc.telemetry
            time.sleep(0.05)

    def start(self):
        try:
            # Use heartbeat_type='B' for extended data including G-forces (surge, sway, heave)
            self.tc = TurismoClient(ps_ip=self.ps5_ip, heartbeat_type='B')
            self.tc.start()
            print(f"✅ Telemetry started (connected to {self.ps5_ip}).")
        except PlayStationOnStandbyError:
            print("❗ PS5 is asleep—wake it up.")
            return
        except PlayStationNotFoundError:
            print("❗ PS5 not found on LAN.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.tc:
            try:
                self.tc.stop()
            except Exception:
                pass
        print("🛑 Telemetry stopped.")

    def get_latest(self):
        return self.latest