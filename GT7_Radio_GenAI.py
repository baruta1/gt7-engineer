import nest_asyncio
nest_asyncio.apply()

import os
import sys
import asyncio
import discord
from discord.ext import commands, voice_recv
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI
# using faster_whisper instead of whisper
from discord import FFmpegPCMAudio
from pydub import AudioSegment, effects
import aiohttp
import async_timeout
from telemetry_server import TelemetryServer
from pydub.effects import low_pass_filter, high_pass_filter
from pydub.generators import WhiteNoise
import edge_tts
from gt_telem import TurismoClient
from gt_telem.events import GameEvents, RaceEvents
import re
import re
import inflect
from faster_whisper import WhisperModel
import os
import time
import subprocess
from pathlib import Path

import logging
logging.getLogger("discord").setLevel(logging.WARNING)
logging.getLogger("discord.ext.voice_recv").setLevel(logging.WARNING)
logging.getLogger("paramiko").setLevel(logging.WARNING)

# allow duplicate OpenMP runtimes (unsafe, but gets you running)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Initialize inflect engine
p = inflect.engine()


# Driver Name - can be set via !setdriver command or DRIVER_NAME in .env
driver_name = ""
TRIGGER_PHRASE = "radio|really|video"

def get_main_prompt():
    """Generate the main prompt with current driver name."""
    name = driver_name if driver_name else "Driver"
    return (
        f"You are Aoife, a sharp and witty Irish-British female Formula 1 race engineer for {name}. "
        f"You're their trusted mate on the pit wall - quick with a joke but deadly serious when it matters. "
        f"You have a warm Dublin accent in your phrasing, occasionally dropping in Irish expressions like 'grand', 'brilliant', 'right so'. "
        f"You're supremely competent - you know your data cold and give precise, actionable advice. "
        f"You keep messages SHORT and punchy - this is racing, not a chat. Max 1-2 sentences. "
        f"You're supportive but not sycophantic - if {name} messes up, you'll note it with dry humor then move on. "
        f"CRITICAL RULES: "
        f"- ONLY discuss racing, car setup, strategy, lap times, tyres, fuel, weather, positions. "
        f"- NEVER discuss anything unrelated to the current race or motorsport. "
        f"- If asked about non-racing topics, deflect with humor and refocus: 'Focus on the race, we'll chat after!' "
        f"- NO emojis, NO symbols, NO asterisks. Just clean spoken text. "
        f"- Always end with proper punctuation. Be concise and complete."
    )


class QuietFFmpegPCMAudio(FFmpegPCMAudio):
    def __init__(self, source, **kwargs):
        if 'before_options' not in kwargs:
            kwargs['before_options'] = ''
        if 'options' not in kwargs:
            kwargs['options'] = '-vn'

        # Inject creationflags to suppress console window on Windows (only on Windows)
        if sys.platform == 'win32':
            self.creationflags = subprocess.CREATE_NO_WINDOW

        super().__init__(source, **kwargs)
        
def apply_radio_filter(audio: AudioSegment) -> AudioSegment:
    # Step 1: Extremely narrow bandpass to simulate walkie-talkie frequency
    filtered = high_pass_filter(audio, cutoff=500)
    filtered = low_pass_filter(filtered, cutoff=3000)

    # Step 2: Add brutal compression to flatten dynamics
    filtered = effects.compress_dynamic_range(filtered, threshold=-30.0, ratio=10.0)

    # Step 3: Bitcrush simulation (reduce bitrate fidelity)
    filtered = filtered.set_sample_width(1)  # reduce to 8-bit depth

    # Step 4: Add stronger static for a gritty texture
    noise = WhiteNoise().to_audio_segment(duration=len(filtered), volume=-60)
    filtered = filtered.overlay(noise)

    # Step 5: Apply band-limited EQ feel (resample to 8000 Hz)
    filtered = filtered.set_frame_rate(8000).set_channels(1)
    filtered = effects.normalize(filtered)

    return filtered


def purge_old_recordings(older_than_sec=600):
    import glob
    now = time.time()
    for p in glob.glob("recordings/*.wav"):
        try:
            if now - os.path.getmtime(p) > older_than_sec:
                os.remove(p)
        except Exception:
            pass
        
def safe_stop_listening(vc):
    try:
        vc.stop_listening()
    except Exception:
        pass
    
# ─── Setup ─────────────────────────────────────────────
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AUTO_CHANNEL_ID = os.getenv("AUTO_CHANNEL_ID")
DRIVER_USER_ID = os.getenv("DRIVER_USER_ID")

# Load driver name from .env if set
driver_name = os.getenv("DRIVER_NAME", "")
if driver_name:
    print(f"👤 Driver name loaded from .env: {driver_name}")

client = OpenAI(api_key=OPENAI_API_KEY)

#model = whisper.load_model("small")
model = WhisperModel("tiny", compute_type='int8')

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

os.makedirs("recordings", exist_ok=True)
os.makedirs("tts", exist_ok=True)


# ___ Radio ________________
#telemetry = TelemetryServer()
#telemetry.start()
prev_lap = None
has_initialized_position = False
last_fuel_alert = 100
prev_position = None          # for overtake detection
last_pos_call = 0             # throttle radio spam (seconds)
prev_best_ms  = None          # track best‑lap improvement
race_started = False    # becomes True on on_race_start
radio_paused = True     # start muted until race actually begins
voice_conn   = None     # populated once we have the Discord VC
announced_fuel_levels = set()
watchdog_snooze_until = 0.0  # epoch seconds until which the watchdog is disabled

# ─── NEW: Tire Wear Tracking ───────────────────────────────
initial_tire_radius = None        # Store tire radius at race start
tire_wear_announced = set()       # Track which wear thresholds announced
last_tire_wear_call = 0           # Cooldown for tire announcements

# ─── NEW: Smart Lap Delta ──────────────────────────────────
last_lap_delta_call = 0           # Cooldown for lap delta announcements
consecutive_slow_laps = 0         # Track if driver is struggling

# ─── NEW: Incident Detection ───────────────────────────────
last_incident_call = 0            # Cooldown for incident announcements
prev_g_forces = None              # Track previous G-forces for delta

# ─── NEW: Driving Style Analysis ───────────────────────────
tcs_activations = 0               # Count TCS activations per lap
asm_activations = 0               # Count ASM activations per lap
wheelspin_events = 0              # Count wheelspin events per lap
lockup_events = 0                 # Count lockup events per lap
last_driving_style_call = 0       # Cooldown for driving style feedback
prev_tcs_state = False            # Previous TCS state
prev_asm_state = False            # Previous ASM state

def snooze_watchdog(seconds=30):
    """Temporarily disable the 'no audio' watchdog for a grace period."""
    global watchdog_snooze_until
    watchdog_snooze_until = time.time() + seconds
    
# ─── TTS and Transcription ─────────────────────────────

# Radio sound effects
# Confirmation beep - played when driver says "Radio" to confirm receipt
confirmation_sound = "Message_Confirmation.wav"
# Radio on/off sounds
radio_on = AudioSegment.from_file("Radio/Start.flac").set_channels(1).apply_gain(-5)
radio_off = AudioSegment.from_file("Radio/End1.wav").set_channels(1).apply_gain(-5)
# F1 "doo-doo-doo" - played at START of automated/unprompted messages to get attention
f1_alert = AudioSegment.from_file("Radio/f1_end.mp3").set_channels(1).apply_gain(-3)

# Piper TTS Voice (British female - alba, local and fast)
from piper import PiperVoice
from piper.config import SynthesisConfig
import wave
PIPER_MODEL_PATH = Path(__file__).parent / "piper_models" / "en_GB-alba-medium.onnx"
piper_voice = PiperVoice.load(str(PIPER_MODEL_PATH))
# Speed: length_scale < 1 = faster, > 1 = slower (0.9 = 10% faster, still clear)
piper_config = SynthesisConfig(length_scale=0.9, noise_scale=0.5, noise_w_scale=0.5)


was_in_race = False
paused_since = 0.0

last_rx_time = 0.0

RECORD_DIR = Path("recordings")
RECORD_DIR.mkdir(exist_ok=True)

class UserFilterSink(voice_recv.AudioSink):
    def __init__(self, inner, allow_ids):
        self.inner = inner
        self.allow = set(allow_ids)

    def wants_opus(self):
        return self.inner.wants_opus()

    def write(self, user, data):
        # mark that we *are* receiving any audio at all
        global last_rx_time
        last_rx_time = time.time()

        # PS5 bridge often leaves user==None → pass it through
        if user is None:
            self.inner.write(user, data)
            return

        # If the user id is known, gate it
        if getattr(user, "id", None) in self.allow:
            self.inner.write(user, data)

    def cleanup(self):
        self.inner.cleanup()
        
        
        
async def record_one(vc, seconds=5, user_id=None):
    path = RECORD_DIR / f"idle_{int(time.time()*1000)}.wav"
    done = asyncio.Event()
    loop = asyncio.get_running_loop()  # capture main loop here

    sink = voice_recv.WaveSink(str(path))
    if user_id is not None:
        try:
            sink = voice_recv.ConditionalFilter.user_filter([user_id]).pipe(sink)
        except AttributeError:
            sink = UserFilterSink(sink, [user_id])
        
    def _after(exc):
        # runs in a worker thread; notify the main loop safely
        loop.call_soon_threadsafe(done.set)

    vc.listen(sink, after=_after)
    try:
        await asyncio.sleep(seconds)
    finally:
        vc.stop_listening()
        try:
            await asyncio.wait_for(done.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass

    if not path.exists() or path.stat().st_size < 2000:
        try:
            path.unlink()  # delete tiny/empty files so they don't accumulate
        except Exception:
            pass
        return None
    return str(path)

def _piper_synthesize(text, wav_path):
    """Synchronous Piper TTS synthesis with speed config."""
    with wave.open(wav_path, 'wb') as f:
        piper_voice.synthesize_wav(text, f, syn_config=piper_config)

async def synthesize_response(text, filename="engineer", is_automated=False, retries=3, timeout_sec=30):
    """
    Generate TTS with F1-style radio sounds using Piper (local, fast).

    For AUTOMATED messages (unprompted - lap updates, position changes):
        [f1_alert doo-doo-doo] + [radio_on crackle] + [speech] + [radio_off]

    For RESPONSE messages (replies to driver's "Radio" questions):
        [radio_on crackle] + [speech] + [radio_off]
    """
    text = re.sub(r"P(\d+)", replace_p_with_words, text)
    wav_path = f"tts/{filename}_raw.wav"
    path = f"tts/{filename}.mp3"

    for attempt in range(retries):
        try:
            print(f"🗣️ TTS attempt {attempt + 1}...")

            # Run Piper in thread executor (it's synchronous)
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, _piper_synthesize, text, wav_path),
                timeout=timeout_sec
            )

            if not os.path.exists(wav_path):
                raise FileNotFoundError("TTS output file not found after generation.")

            # Apply radio filter to voice
            audio = AudioSegment.from_file(wav_path)
            audio = audio.set_channels(1)
            radio_voice = apply_radio_filter(audio)

            # Build radio sequence based on message type
            if is_automated:
                # Automated: [f1_alert] + [speech] + [radio_off] (no beep after F1 sound)
                radio_audio = f1_alert + radio_voice + radio_off
            else:
                # Response: [radio_on] + [speech] + [radio_off]
                radio_audio = radio_on + radio_voice + radio_off

            radio_audio.export(path, format="mp3")

            # Clean up raw file
            try:
                os.remove(wav_path)
            except:
                pass

            return path

        except (aiohttp.ClientConnectorError, FileNotFoundError, TimeoutError, Exception) as e:
            print(f"❌ TTS attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(0.5)

    print("❌ All TTS attempts failed. No audio response will be played.")
    return None

    
async def transcribe_audio(path):
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        print(f"⚠️ Skipping empty or invalid audio file: {path}")
        return ""

    try:
        loop = asyncio.get_event_loop()
        #result = await loop.run_in_executor(None, lambda: model.transcribe(path))
        #return result["text"].strip().lower()
        #return result["text"].strip().lower()
        segments, info = await loop.run_in_executor(None, lambda: model.transcribe(path, beam_size=5, language="en"))
        text = " ".join([seg.text for seg in segments])
        return text.strip().lower()
    except Exception as e:
        print(f"❌ Failed to transcribe {path}: {e}")
        return ""
    



TRIGGER_RE = re.compile(r'^(?:radio|really|video)\b', re.I)

def starts_with_trigger(text: str) -> bool:
    return bool(TRIGGER_RE.match(text or ""))


def ms_to_min_sec(ms: int) -> tuple[int, int]:
    """Return (minutes, seconds) for a duration given in milliseconds."""
    seconds_total = ms // 1000               # drop the milliseconds part
    minutes, seconds = divmod(seconds_total, 60)
    return (f"{minutes} minutes and {seconds} seconds")  


def to_milliseconds(time_str: str) -> int:
    """
    Convert strings like '3 minutes and 17 seconds' to milliseconds.
    """
    # 1. Pull out the numbers with a simple regex
    match = re.fullmatch(r'\s*(\d+)\s+minutes?\s+and\s+(\d+)\s+seconds?\s*', time_str)
    if not match:
        raise ValueError(f"Unsupported format: {time_str}")
    
    minutes, seconds = map(int, match.groups())

    # 2. Convert to ms
    return (minutes * 60 + seconds) * 1000
#  Latest telemetry data:
def latest_telemetry_data():
    t = telemetry.get_latest()
    if not t:
        return None

    curr_lap    = t.current_lap
    total_laps  = t.total_laps
    last_lap_ms = t.last_lap_time_ms

    return {
        "current_lap": curr_lap,
        "position": t.race_start_pos,
        "total_laps": total_laps,
        "lap_time_ms": ms_to_min_sec(last_lap_ms),
        "fuel_pct": (t.fuel_level / t.fuel_capacity) * 100,
        "speed_kph": t.speed_mps * 3.6,
        "engine_rpm": int(t.engine_rpm),
        "total_cars": t.total_cars,
        "tire_fl_temp": t.tire_fl_temp,
        "tire_fr_temp": t.tire_fr_temp,
        "tire_rl_temp": t.tire_rl_temp,
        "tire_rr_temp": t.tire_rr_temp,
        "oil_pressure": t.oil_pressure,
        "water_temp": t.water_temp,
        "oil_temp": t.oil_temp,
        "best_lap_time_ms": ms_to_min_sec(t.best_lap_time_ms),   # NEW
    }



        
async def play_line(vc, text, cache_tag, is_automated=True):
    """TTS + play, but only if radio is 'live'. Also stop listening to avoid echo.

    is_automated=True for unprompted messages (lap updates, position changes).
    These get the F1 alert sound at the start.
    """
    global voice_conn

    if radio_paused:
        print(f"⚠️ play_line skipped ({cache_tag}): radio_paused=True")
        return
    if not race_started:
        print(f"⚠️ play_line skipped ({cache_tag}): race not started")
        return

    # Use the global voice_conn which gets refreshed on gateway reconnect
    vc = voice_conn if voice_conn else vc

    if not vc or not vc.is_connected():
        print(f"⚠️ play_line skipped ({cache_tag}): vc not connected")
        return

    try:
        print(f"🎙️ play_line: generating TTS for '{cache_tag}'...")
        mp3 = await synthesize_response(text, filename=cache_tag, is_automated=is_automated)
        if mp3:
            # Snooze while we intentionally create silence (stop listening + TTS)
            snooze_watchdog(30)
            safe_stop_listening(vc)  # avoid echo
            # Reset the "no_rx_for" clock so the watchdog won't misfire
            global last_rx_time
            last_rx_time = time.time()

            # Re-check voice_conn in case it was refreshed during TTS generation
            if voice_conn and voice_conn != vc and voice_conn.is_connected():
                print(f"🔄 Using refreshed voice connection for '{cache_tag}'")
                vc = voice_conn

            # Verify voice connection is still good before playing
            if not vc.is_connected():
                print(f"⚠️ Voice disconnected before playing '{cache_tag}' - reconnecting...")
                try:
                    ch = vc.channel
                    await asyncio.sleep(0.5)  # Brief pause before reconnecting
                    vc = await ch.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False, self_mute=False)
                    voice_conn = vc
                except Exception as e:
                    print(f"❌ Reconnect failed: {e}")
                    return

            print(f"🔊 play_line: playing '{cache_tag}'...")
            vc.play(QuietFFmpegPCMAudio(mp3))
            while vc.is_playing():
                await asyncio.sleep(0.1)
            print(f"✅ play_line: finished '{cache_tag}'")

            # Reset voice refresh timer after successful play
            global last_voice_refresh
            last_voice_refresh = time.time()

            # Small delay after playing to let Discord audio stream reset
            await asyncio.sleep(0.5)

            # Add a short tail so we don't immediately re-arm the watchdog on the next loop tick
            snooze_watchdog(10)
            last_rx_time = time.time()
        else:
            print(f"⚠️ play_line: TTS returned None for '{cache_tag}'")
    except Exception as e:
        print(f"Radio line failed ({cache_tag}): {e}")


def replace_p_with_words(match):
    number = int(match.group(1))
    word = p.number_to_words(number)
    return f"pee {word}"
   

# ─── helper for overtakes ──────────────────────────────────
async def maybe_handle_overtake(vc, latest):
    global prev_position, last_pos_call, has_initialized_position
    
    curr_pos = latest.race_start_pos


    # Don't check for overtakes until after lap 2 and position is initialized
    if not has_initialized_position:
        if latest.current_lap >= 2:
            prev_position = curr_pos
            has_initialized_position = True
        return            
            # only after lap 2, with an 60s cooldown
    if latest.current_lap >= 2 and prev_position is not None and curr_pos != prev_position:
        if time.time() - last_pos_call > 45:
            # build your prompt
            flavour = "gained a place" if curr_pos < prev_position else "lost a place"
            stats = latest_telemetry_data()
            prompt =get_main_prompt()+ (f" The driver just {flavour} "
                f"(from P{prev_position} to P{curr_pos}). In 10 words or fewer, quip about it. Stats for the car are: {stats}"
                f"Be sure to also mention what place they are now in, like P one (please leave a space between the P and the number of their place). Keep things short, no more than 10 words total!"
            )
            msg = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}]
            ).choices[0].message.content
            await play_line(vc, msg, "overtake")
            last_pos_call = time.time()

    prev_position = curr_pos
    
# ─── helper for lap updates ───────────────────────────────
async def maybe_handle_lap_update(vc, latest):
    global prev_lap
    global prev_best_ms

    if not latest:
        return

    lap = latest.current_lap
    total = latest.total_laps

    # Skip invalid states
    if lap is None or lap == 1:
        return

    # Guard against telemetry reset (lap going backwards or to 0)
    if lap == 0 or total == 0:
        print(f"⚠️ Ignoring invalid lap data: lap={lap}, total={total}")
        return

    if prev_lap is not None and lap < prev_lap:
        print(f"⚠️ Ignoring backwards lap change: {prev_lap} -> {lap}")
        return

    if lap == prev_lap:
        return

    print(f"🏎️ LAP CHANGE: {prev_lap} -> {lap} (of {total}) - generating update...")
    stats = latest_telemetry_data()

    # Determine lap context
    laps_remaining = total - lap if total > 0 else 0

    if total > 0 and lap > total:
        # Race finished (crossed line on final lap)
        prompt = get_main_prompt() + (
            f" The car stats are: {stats}. "
            f"The race just FINISHED! Congratulate the driver. "
            f"Give a super-short celebratory message about their final position."
        )
    elif total > 0 and lap == total:
        # LAST LAP - make it clear!
        prompt = get_main_prompt() + (
            f" The car stats are: {stats}. "
            f"This is the FINAL LAP! Lap {lap} of {total}. "
            f"Give an urgent, short message - push to the end! Mention it's the last lap."
        )
    elif total > 0 and lap == total - 1:
        # Second to last lap
        prompt = get_main_prompt() + (
            f" The car stats are: {stats}. "
            f"Lap {lap} of {total} - just ONE lap to go after this! "
            f"Give a short update, mention there's one lap remaining."
        )
    else:
        # Normal lap update
        prompt = get_main_prompt() + (
            f" The car stats are: {stats}. "
            f"We're on lap {lap} of {total} ({laps_remaining} laps remaining). "
            f"Give a one or two sentence update."
        )
    if latest and latest.best_lap_time_ms != -1:
        if prev_best_ms is None or latest.best_lap_time_ms < to_milliseconds( prev_best_ms):
            prev_best_ms = stats["best_lap_time_ms"]
            stats = latest_telemetry_data()
            prompt += f" And please tell the driver they just set a new best lap of {prev_best_ms} ms. "
    summary = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}]
    ).choices[0].message.content
    print(f"🏎️ Lap {lap} update: {summary}")
    await play_line(vc, summary, "lap_update")
    prev_lap = lap

# ─── NEW: Tire Wear Detection ──────────────────────────────
async def maybe_handle_tire_wear(vc, latest):
    """Track tire wear by monitoring radius decrease. Announce at significant thresholds."""
    global initial_tire_radius, tire_wear_announced, last_tire_wear_call

    if not latest or not race_started:
        return

    # Get current tire radii
    current_radii = {
        'fl': latest.tire_fl_radius,
        'fr': latest.tire_fr_radius,
        'rl': latest.tire_rl_radius,
        'rr': latest.tire_rr_radius
    }

    # Initialize on first call after race start
    if initial_tire_radius is None:
        initial_tire_radius = current_radii.copy()
        return

    # Calculate wear percentage for each tire (radius decrease = wear)
    wear = {}
    for pos in ['fl', 'fr', 'rl', 'rr']:
        if initial_tire_radius[pos] > 0:
            wear[pos] = ((initial_tire_radius[pos] - current_radii[pos]) / initial_tire_radius[pos]) * 100
        else:
            wear[pos] = 0

    # Get worst tire wear
    max_wear = max(wear.values())
    worst_tire = max(wear, key=wear.get)
    tire_names = {'fl': 'front left', 'fr': 'front right', 'rl': 'rear left', 'rr': 'rear right'}

    # Announce at thresholds: 15%, 25%, 40% wear (sparing announcements)
    # 90 second cooldown between tire wear announcements
    thresholds = [15, 25, 40]
    for thresh in thresholds:
        if max_wear >= thresh and thresh not in tire_wear_announced:
            if time.time() - last_tire_wear_call > 90:
                tire_wear_announced.add(thresh)
                last_tire_wear_call = time.time()

                if thresh >= 40:
                    severity = "getting worn now"
                elif thresh >= 25:
                    severity = "showing some wear"
                else:
                    severity = "starting to go off"

                prompt = get_main_prompt() + (
                    f" The {tire_names[worst_tire]} tyre is {severity} - about {int(max_wear)}% worn. "
                    f"Give a very brief (under 12 words) update about tyre condition. "
                    f"Be practical - should they push or conserve?"
                )
                msg = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": prompt}]
                ).choices[0].message.content
                await play_line(vc, msg, f"tire_wear_{thresh}")
                break

# ─── NEW: Smart Lap Delta ──────────────────────────────────
async def maybe_handle_lap_delta(vc, latest):
    """Announce lap time delta vs personal best, but only when meaningful."""
    global last_lap_delta_call, consecutive_slow_laps, prev_best_ms

    if not latest or not race_started:
        return

    # Only check after lap 2 (lap 1 is often messy start)
    if latest.current_lap < 2:
        return

    # Need valid lap times
    if latest.last_lap_time_ms <= 0 or latest.best_lap_time_ms <= 0:
        return

    # 60 second cooldown between lap delta announcements
    if time.time() - last_lap_delta_call < 60:
        return

    last_lap = latest.last_lap_time_ms
    best_lap = latest.best_lap_time_ms
    delta_ms = last_lap - best_lap
    delta_sec = delta_ms / 1000.0

    # Determine if worth announcing
    msg = None

    if delta_ms == 0:
        # They just set a new PB! (handled in lap_update, skip here)
        consecutive_slow_laps = 0
        return
    elif delta_ms < 500:  # Within 0.5s of best - good lap!
        consecutive_slow_laps = 0
        prompt = get_main_prompt() + (
            f" Driver just did a lap only {delta_sec:.1f} seconds off their best. "
            f"Give quick encouragement (under 10 words). They're on pace!"
        )
        msg = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}]
        ).choices[0].message.content
    elif delta_ms > 3000:  # More than 3s off - something happened
        consecutive_slow_laps += 1
        # Only mention if it's not a pattern (avoid nagging)
        if consecutive_slow_laps == 1:
            prompt = get_main_prompt() + (
                f" Driver's last lap was {delta_sec:.1f} seconds off their best. "
                f"Something might have happened. Quick check-in (under 10 words), "
                f"but don't be negative - maybe traffic or a moment."
            )
            msg = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}]
            ).choices[0].message.content
    else:
        consecutive_slow_laps = 0

    if msg:
        last_lap_delta_call = time.time()
        await play_line(vc, msg, "lap_delta")

# ─── NEW: Incident Detection ───────────────────────────────
async def maybe_handle_incident(vc, latest):
    """Detect big impacts or slides using G-force data."""
    global last_incident_call, prev_g_forces

    if not latest or not race_started:
        return

    # Get G-force data (surge = forward/back, sway = left/right, heave = up/down)
    surge = getattr(latest, 'surge', None)
    sway = getattr(latest, 'sway', None)
    heave = getattr(latest, 'heave', None)

    # Skip if motion data not available
    if surge is None or sway is None:
        return

    # 45 second cooldown between incident calls
    if time.time() - last_incident_call < 45:
        return

    # Calculate total G-force magnitude
    total_g = (surge**2 + sway**2) ** 0.5

    # Detect significant lateral G (big slide or impact)
    # Thresholds tuned for notable events only
    if abs(sway) > 2.5:  # Big lateral hit or slide
        last_incident_call = time.time()
        prompt = get_main_prompt() + (
            f" The car just had a significant lateral load - possibly a slide or contact. "
            f"Quick check-in under 8 words. Be supportive, not alarming."
        )
        msg = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}]
        ).choices[0].message.content
        await play_line(vc, msg, "incident_lateral")
    elif abs(surge) > 3.0:  # Big decel - hard braking or impact
        # Only flag if unexpected (could be normal hard braking zone)
        if latest.brake < 200:  # Not heavy braking, so likely an impact
            last_incident_call = time.time()
            prompt = get_main_prompt() + (
                f" Car experienced sudden deceleration without heavy braking - possible contact. "
                f"Brief check-in under 8 words."
            )
            msg = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}]
            ).choices[0].message.content
            await play_line(vc, msg, "incident_impact")

# ─── NEW: Driving Style Analysis ───────────────────────────
async def update_driving_style_counters(latest):
    """Track TCS/ASM activations and wheel events per lap. Called every loop."""
    global tcs_activations, asm_activations, wheelspin_events, lockup_events
    global prev_tcs_state, prev_asm_state

    if not latest or not race_started:
        return

    # Track TCS activations (rising edge)
    tcs_active = getattr(latest, 'tcs_active', False)
    if tcs_active and not prev_tcs_state:
        tcs_activations += 1
    prev_tcs_state = tcs_active

    # Track ASM activations (rising edge)
    asm_active = getattr(latest, 'asm_active', False)
    if asm_active and not prev_asm_state:
        asm_activations += 1
    prev_asm_state = asm_active

    # Detect wheelspin (rear wheels spinning faster than fronts significantly)
    try:
        front_avg_rps = (latest.wheel_fl_rps + latest.wheel_fr_rps) / 2
        rear_avg_rps = (latest.wheel_rl_rps + latest.wheel_rr_rps) / 2
        if front_avg_rps > 0 and rear_avg_rps / front_avg_rps > 1.15:  # 15% faster = wheelspin
            wheelspin_events += 1
    except (AttributeError, ZeroDivisionError):
        pass

    # Detect lockups (one wheel much slower than others under braking)
    try:
        if latest.brake > 150:  # Only during braking
            wheels = [latest.wheel_fl_rps, latest.wheel_fr_rps, latest.wheel_rl_rps, latest.wheel_rr_rps]
            avg_rps = sum(wheels) / 4
            if avg_rps > 0:
                for w in wheels:
                    if w < avg_rps * 0.7:  # One wheel 30% slower = lockup
                        lockup_events += 1
                        break
    except (AttributeError, ZeroDivisionError):
        pass

async def maybe_handle_driving_style(vc, latest):
    """Give feedback on driving style at end of lap, but only if noteworthy."""
    global tcs_activations, asm_activations, wheelspin_events, lockup_events
    global last_driving_style_call, prev_lap

    if not latest or not race_started:
        return

    # Only announce on lap change, and not every lap (every 3rd lap max)
    current_lap = latest.current_lap
    if current_lap < 3 or current_lap == prev_lap:
        return

    # 120 second cooldown (roughly every 2-3 laps)
    if time.time() - last_driving_style_call < 120:
        # Reset counters for next lap anyway
        tcs_activations = asm_activations = wheelspin_events = lockup_events = 0
        return

    # Check if there's something worth mentioning
    msg = None

    if tcs_activations > 15:
        prompt = get_main_prompt() + (
            f" TCS activated {tcs_activations} times last lap. "
            f"Brief driving tip under 12 words - maybe ease throttle on exits."
        )
        msg = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}]
        ).choices[0].message.content
    elif lockup_events > 8:
        prompt = get_main_prompt() + (
            f" Detected {lockup_events} possible lockups last lap. "
            f"Brief tip under 12 words about braking."
        )
        msg = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}]
        ).choices[0].message.content
    elif wheelspin_events > 20:
        prompt = get_main_prompt() + (
            f" Lots of wheelspin detected ({wheelspin_events} events). "
            f"Brief throttle management tip under 12 words."
        )
        msg = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}]
        ).choices[0].message.content

    if msg:
        last_driving_style_call = time.time()
        await play_line(vc, msg, "driving_style")

    # Reset counters for next lap
    tcs_activations = asm_activations = wheelspin_events = lockup_events = 0

# ─── Core Voice Handling ───────────────────────────────

async def handle_engineer_flow(vc, driver_user_id):
    """
    • Radio stays muted until we *know* the race has begun (lap==0 → start).
    • While paused/loading/menus, radio is muted and playback is stopped.
    • Debounced race-finish detection avoids false “finish” spam.
    """
    global race_started, radio_paused
    global prev_lap, prev_position, prev_best_ms
    global last_fuel_alert, last_pos_call
    global was_in_race, paused_since
    global last_rx_time
    loop_count = 0

    while vc.is_connected():
        # Periodic cleanup (once a minute of loop iterations)
        if loop_count % 60 == 0:
            purge_old_recordings()
        loop_count += 1

        # Periodic voice refresh to prevent stale connections
        global last_voice_refresh
        now = time.time()
        if (race_started
            and not vc.is_playing()
            and (now - last_voice_refresh) > VOICE_REFRESH_INTERVAL):
            print("🔄 Periodic voice refresh...")
            try:
                ch = vc.channel
                await vc.disconnect(force=True)
                await asyncio.sleep(0.3)
                vc = await ch.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False, self_mute=False)
                voice_conn = vc
                last_voice_refresh = now
                print("✅ Voice refreshed")
            except Exception as e:
                print(f"⚠️ Voice refresh failed: {e}")

        t = telemetry.get_latest()

        # Compute paused/menu/loading in a null-safe way
        in_menu_or_loading = (
            (t is None)
            or getattr(t, "is_loading", False)
            or getattr(t, "cars_on_track", 0) == 0
        )
        now = time.time()
        no_rx_for = now - (last_rx_time or now)
        snoozed = now < watchdog_snooze_until
        
        # Watchdog: only refresh/reconnect if voice connection seems dead
        # Increased timeouts - driver won't always be speaking during race
        if (not snoozed
            and not in_menu_or_loading
            and race_started
            and not vc.is_playing()
            and no_rx_for > 60):  # Increased from 10s to 60s
            print("🩺 No incoming audio for >60s — refreshing voice listener")
            safe_stop_listening(vc)
            last_rx_time = time.time()  # Reset timer after refresh
            await asyncio.sleep(0.25)

        if (not snoozed
            and not in_menu_or_loading
            and race_started
            and not vc.is_playing()
            and no_rx_for > 120):  # Increased from 30s to 120s
            print("🩺 No audio for >120s — reconnecting voice")
            try:
                ch = vc.channel
                await vc.disconnect(force=True)
                vc = await ch.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False, self_mute=False)
                voice_conn = vc
                last_rx_time = time.time()
            except Exception as e:
                print("Reconnect failed:", e)


        # If we ever saw a lap and telemetry "rewound" total_laps, treat as paused
        if (prev_lap is not None) and (t is not None):
            try:
                if (t.total_laps != 0) and (t.total_laps < prev_lap):
                    in_menu_or_loading = True
            except AttributeError:
                pass

        # Rising edge: first valid on-track telemetry sets race started (lap==0)
        if (not in_menu_or_loading) and (not race_started) and (t is not None) and (t.current_lap == 0):
            race_started = True
            radio_paused = False
            was_in_race = True
            print("🟢 on_in_race detected – radio live")
            await play_line(vc, "Engineer here — radio check, good luck out there!", "intro_start")

        # Debounce the finish: require sustained paused/menu/loading for >2.5s
        now = time.time()
        if in_menu_or_loading:
            if paused_since == 0.0:
                paused_since = now
        else:
            paused_since = 0.0

        if was_in_race and in_menu_or_loading and paused_since and (now - paused_since) > 2.5:
            await play_line(vc, "Solid stint – see you in the garage, we'll debrief later.", "race_finish")
            race_started = False
            radio_paused = True
            was_in_race = False
            prev_lap = prev_position = prev_best_ms = None
            last_fuel_alert = 100
            announced_fuel_levels.clear()

            # Reset new tracking variables
            global initial_tire_radius, tire_wear_announced, last_tire_wear_call
            global last_lap_delta_call, consecutive_slow_laps
            global last_incident_call, prev_g_forces
            global tcs_activations, asm_activations, wheelspin_events, lockup_events
            global last_driving_style_call, prev_tcs_state, prev_asm_state

            initial_tire_radius = None
            tire_wear_announced.clear()
            last_tire_wear_call = 0
            last_lap_delta_call = 0
            consecutive_slow_laps = 0
            last_incident_call = 0
            prev_g_forces = None
            tcs_activations = asm_activations = wheelspin_events = lockup_events = 0
            last_driving_style_call = 0
            prev_tcs_state = prev_asm_state = False

            print("🏁 Race finished – radio reset")
            await asyncio.sleep(1.0)
            continue

        # Entering pause/menu while live → mute
        if in_menu_or_loading and not radio_paused:
            radio_paused = True
            if vc.is_playing():
                vc.stop()
            print("🔇 Radio muted (sim paused / menu)")

        # Leaving pause/menu after starting → unmute
        if (not in_menu_or_loading) and radio_paused and race_started:
            radio_paused = False
            print("🎙️ Radio live again")

        # If we’re not live, idle a bit
        if radio_paused or not race_started:
            await asyncio.sleep(1)
            continue

        # Never record while playing (prevents echo and file locks)
        if vc.is_playing():
            await asyncio.sleep(0.1)
            continue

        # Record clip (unique filename; filtered to the driver if supported)
        audio_path = await record_one(vc, seconds=5, user_id=driver_user_id)
        if not audio_path:
            # Still run telemetry checks even if no audio recorded
            latest = telemetry.get_latest()
            if latest:
                await maybe_handle_overtake(vc, latest)
                await maybe_handle_lap_update(vc, latest)
                await maybe_announce_fuel(vc)
                await update_driving_style_counters(latest)
                await maybe_handle_tire_wear(vc, latest)
                await maybe_handle_lap_delta(vc, latest)
                await maybe_handle_incident(vc, latest)
                await maybe_handle_driving_style(vc, latest)
            continue

        try:
            user_text = await transcribe_audio(audio_path)
        finally:
            try:
                os.remove(audio_path)
            except Exception:
                pass  # sweeper will catch any stragglers
        #user_text = await transcribe_audio(audio_path)
        #try:
        #    os.remove(audio_path)
        #except PermissionError:
        #    pass  # if a handle lingers, the sweeper will catch it later
        print("🗣️ You said:", user_text)

        latest = telemetry.get_latest()

        # Triggered, short interaction
        if starts_with_trigger(user_text):
            # stop listening before any playback
            safe_stop_listening(vc)
            
            
            # Snooze the watchdog for the beep
            snooze_watchdog(10)
            last_rx_time = time.time()
            
            
            vc.play(QuietFFmpegPCMAudio(confirmation_sound, options="-filter:a volume=0.2"))
            while vc.is_playing():
                await asyncio.sleep(0.2)

            # Clean the trigger word safely
            query = TRIGGER_RE.sub("", user_text.strip().lower(), count=1).strip()
            user_text = ""
            print("🎤 Engineer Triggered Phrase:", query)

            stats = latest_telemetry_data()
            system_prompt = get_main_prompt() + (
                f" The car stats now in the race are as follows: {stats} "
                f"Answer the driver's question as if on the radio. Be as short and concise as possible! Time and speed matter. "
                f"Their question: {query}"
            )
            if not query:
                reply = "Loud and clear. Standing by for your next instruction."
            else:
                reply = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                ).choices[0].message.content

            print("🤖 Engineer:", reply)

            try:
                tts_path = await synthesize_response(reply, "engineer_response")
                if tts_path:
                    # Snooze again for the TTS payload (allow longer)
                    snooze_watchdog(30)
                    last_rx_time = time.time()
                
                    safe_stop_listening(vc)  # re-ensure no echo
                    vc.play(QuietFFmpegPCMAudio(tts_path))
                    while vc.is_playing():
                        await asyncio.sleep(0.2)
                
                    # Tail buffer
                    snooze_watchdog(10)
                    last_rx_time = time.time()
            except Exception as e:
                print(f"❌ TTS failed — no audio response. Reason: {e}")
            continue

        # Telemetry-driven radio calls
        if latest:
            # Debug: show current lap info periodically
            if loop_count % 20 == 0:  # Every ~20 loops
                print(f"📊 Lap {latest.current_lap}/{latest.total_laps} | Pos P{latest.race_start_pos} | prev_lap={prev_lap}")

        await maybe_handle_overtake(vc, latest)
        await maybe_handle_lap_update(vc, latest)
        await maybe_announce_fuel(vc)

        # NEW: Additional telemetry features
        await update_driving_style_counters(latest)  # Track TCS/wheelspin per lap
        await maybe_handle_tire_wear(vc, latest)     # Tire wear announcements
        await maybe_handle_lap_delta(vc, latest)     # Smart lap time feedback
        await maybe_handle_incident(vc, latest)      # Crash/slide detection
        await maybe_handle_driving_style(vc, latest) # Driving style feedback
        
async def maybe_announce_fuel(vc):
    global announced_fuel_levels
    t = telemetry.get_latest()
    if not t:
        return

    fuel_pct = int((t.fuel_level / t.fuel_capacity) * 100)

    for thresh in list(announced_fuel_levels):
        if fuel_pct > thresh:
            announced_fuel_levels.remove(thresh)
    # Only react at exact thresholds (30, 20, 10) and not if already announced
    critical_levels = [50, 20, 10]
    for level in critical_levels:
        if fuel_pct <= level and level not in announced_fuel_levels:
            announced_fuel_levels.add(level)
            
            stats = latest_telemetry_data()
            system_prompt = get_main_prompt() + (
                f" The car's current data is: {stats}.  "
                f"Fuel just dropped below {fuel_pct}%." 
                f"Tell them their fuel level. Make a short quip, but actually give the driver the percentage of fuel they have left. Keep your response ultra short!"
            )

            reply = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt}
                ]
            ).choices[0].message.content
            
            print(f"⛽ Fuel Alert {level}%: {reply}")
            try:
                tts_path = await synthesize_response(reply, f"fuel_{level}", is_automated=True)
                vc.play(QuietFFmpegPCMAudio(tts_path))
                while vc.is_playing():
                    await asyncio.sleep(0.05)
            except Exception as e:
                print(f"❌ Fuel TTS failed: {e}")
            break  # avoid multiple alerts in the same cycle
                
            
# Track last voice reconnect time for periodic refresh
last_voice_refresh = 0
VOICE_REFRESH_INTERVAL = 45  # seconds


# ─── Bot Commands ──────────────────────────────────────

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

    # Auto-connect to voice channel if configured
    if AUTO_CHANNEL_ID and DRIVER_USER_ID:
        print(f"🔄 Auto-connecting to channel {AUTO_CHANNEL_ID}...")
        try:
            # Use fetch_channel (API call) instead of get_channel (cache) for reliability
            channel = await bot.fetch_channel(int(AUTO_CHANNEL_ID))
            if channel is None:
                print(f"❌ Could not find channel {AUTO_CHANNEL_ID}")
                return

            global voice_conn
            driver_user_id = int(DRIVER_USER_ID)
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False, self_mute=False)
            voice_conn = vc
            print(f"🎧 Engineer auto-connected to {channel.name} — waiting for race start …")

            # Start the engineer flow
            asyncio.create_task(handle_engineer_flow(vc, driver_user_id))
        except Exception as e:
            print(f"❌ Auto-connect failed: {e}")


@bot.command()
async def engineer(ctx):
    if not ctx.author.voice:
        return await ctx.send("❌ Join a voice channel first!")

    global voice_conn
    driver_user_id = ctx.author.id
    channel = ctx.author.voice.channel
    vc = await channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False, self_mute=False)
    voice_conn = vc                       # <- save for event handlers
    await ctx.send("🎧 Engineer connected — waiting for race start …")
    
    
    # DO NOT play intro here; wait for on_race_start
    await handle_engineer_flow(vc, driver_user_id)


@bot.command()
async def setdriver(ctx, *, name: str = None):
    """Set the driver name. Usage: !setdriver YourName"""
    global driver_name
    if not name:
        if driver_name:
            await ctx.send(f"👤 Current driver: **{driver_name}**\nUse `!setdriver YourName` to change it.")
        else:
            await ctx.send("👤 No driver name set. Use `!setdriver YourName` to set it.")
        return

    driver_name = name.strip()
    await ctx.send(f"👤 Driver name set to: **{driver_name}**")
    print(f"👤 Driver name changed to: {driver_name}")


def run_with_telemetry(telemetry_server):
    global telemetry
    telemetry = telemetry_server
        
    bot.run(TOKEN)
