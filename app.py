import os, io, tempfile, uuid, zipfile, urllib.request
import numpy as np
import streamlit as st
import soundfile as sf
import librosa
from typing import Tuple




# ================== CONFIG ==================
SR = 44100
PITCHES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
BEATS_DIR = "./beats"
os.makedirs(BEATS_DIR, exist_ok=True)

# pydub (optional) for robust MP3 decode via ffmpeg
try:
    from pydub import AudioSegment
    HAVE_PYDUB = True
except Exception:
    HAVE_PYDUB = False
    AudioSegment = None  # type: ignore


# ================== FFmpeg on-demand ==================
def ensure_ffmpeg_now() -> bool:
    """Download a portable ffmpeg into .ffmpeg/ and put it on PATH (1st time only)."""
    try:
        if os.system("ffmpeg -version >nul 2>&1") == 0:
            return True
    except Exception:
        pass

    root = os.path.abspath(".ffmpeg")
    os.makedirs(root, exist_ok=True)
    z = os.path.join(root, "ffmpeg.zip")
    try:
        if not os.path.exists(z):
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            urllib.request.urlretrieve(url, z)
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(root)

        # find bin
        bin_dir = None
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if d.startswith("ffmpeg-") and os.path.isdir(p):
                cand = os.path.join(p, "bin")
                if os.path.isfile(os.path.join(cand, "ffmpeg.exe")) or os.path.isfile(os.path.join(cand, "ffmpeg")):
                    bin_dir = cand
                    break
        if bin_dir:
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH","")
            if HAVE_PYDUB and AudioSegment is not None:
                AudioSegment.converter = os.path.join(bin_dir, "ffmpeg.exe" if os.name=="nt" else "ffmpeg")
            return os.system("ffmpeg -version >nul 2>&1") == 0
    except Exception:
        return False
    return False


# ================== AUDIO I/O ==================
def load_audio_robust(src, sr=SR, mono=True) -> Tuple[np.ndarray, int]:
    """src = path or UploadedFile. Try librosa, else pydub/ffmpeg. Returns (float32, sr)."""
    # librosa path/bytes
    try:
        if isinstance(src, (str, os.PathLike)):
            y, _ = librosa.load(src, sr=sr, mono=mono)
        else:
            data = src.read() if hasattr(src, "read") else src
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                tmp.write(data); p = tmp.name
            y, _ = librosa.load(p, sr=sr, mono=mono)
            try: os.remove(p)
            except: pass
        if y is not None and len(y) >= 1024 and np.max(np.abs(y)) > 0:
            return y.astype(np.float32), sr
    except Exception:
        pass

    # pydub fallback (if available)
    if HAVE_PYDUB and AudioSegment is not None:
        try:
            if isinstance(src, (str, os.PathLike)):
                seg = AudioSegment.from_file(src)
            else:
                data = src.read() if hasattr(src, "read") else src
                seg = AudioSegment.from_file(io.BytesIO(data))
            arr = np.array(seg.get_array_of_samples())
            if seg.channels == 2:
                arr = arr.reshape((-1,2)).mean(axis=1)
            y = arr.astype(np.float32)
            if arr.dtype.kind in "iu":
                y /= np.iinfo(arr.dtype).max
            if seg.frame_rate != sr:
                y = librosa.resample(y, orig_sr=seg.frame_rate, target_sr=sr)
            return y.astype(np.float32), sr
        except Exception:
            pass

    return np.zeros(0, dtype=np.float32), sr


def load_valid(beat_path: str):
    """Decode beat; if silent, pull ffmpeg and retry once."""
    y, _ = load_audio_robust(beat_path, sr=SR, mono=True)
    if len(y) >= 1024 and np.max(np.abs(y)) > 0:
        return y
    if ensure_ffmpeg_now():
        y, _ = load_audio_robust(beat_path, sr=SR, mono=True)
        if len(y) >= 1024 and np.max(np.abs(y)) > 0:
            return y
    return None


def wav_bytes(y, sr):
    buf = io.BytesIO(); sf.write(buf, y.astype(np.float32), sr, format="WAV"); buf.seek(0); return buf.read()


# ================== DSP ==================
def detect_bpm(y, sr): 
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr); 
    return float(tempo)

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    pc = int(np.argmax(chroma.mean(axis=1)))
    return PITCHES[pc]

def semitone_distance(a, b):
    if a not in PITCHES or b not in PITCHES: return 0
    i, j = PITCHES.index(a), PITCHES.index(b); d = j - i
    if d > 6: d -= 12
    if d < -6: d += 12
    return d

def time_stretch_safe(y, rate):
    if rate <= 0 or not np.isfinite(rate) or len(y) < 2049: return y
    try: return librosa.effects.time_stretch(y, rate)
    except: return y

def pitch_shift_safe(y, sr, steps):
    if abs(steps) < 0.5 or len(y) < 2049: return y
    try: return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    except: return y

def rms(x): 
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


# ================== UI ==================
st.set_page_config(page_title="BeatSwap — QuickStart", page_icon="⚡", layout="centered")
st.title("⚡ BeatSwap — QuickStart")

exts = {".mp3",".wav",".flac",".m4a",".ogg",".aac",".wma",".aiff",".aif"}
beats = [f for f in sorted(os.listdir(BEATS_DIR)) if os.path.splitext(f)[1].lower() in exts]

c1, c2 = st.columns([2,1])
with c1:
    vocal = st.file_uploader("Upload vocal (WAV/MP3/etc.)", type=None)
    if vocal: st.success(f"Got file: {vocal.name} ({vocal.size/1e6:.2f} MB)")
with c2:
    beat_name = st.selectbox("Pick a beat (no pre-scan)", options=beats or ["(no beats found)"])

match_pitch = st.checkbox("Match key (if known)", value=True)
preview_30s = st.checkbox("30s preview", value=True)

if st.button("Analyze + Remix"):
    if not vocal:
        st.error("Please upload a vocal file."); st.stop()
    if not beats:
        st.error("No beats found in ./beats"); st.stop()

    # ---- VOCAL (with ffmpeg retry) ----
    v, _ = load_audio_robust(vocal, sr=SR, mono=True)
    if preview_30s: v = v[: SR*30]
    if len(v) < 2048 or rms(v) == 0:
        if ensure_ffmpeg_now():
            v, _ = load_audio_robust(vocal, sr=SR, mono=True)
            if preview_30s: v = v[: SR*30]
    if len(v) < 2048 or rms(v) == 0:
        st.error("Vocal undecodable (try WAV)."); st.stop()
    v_bpm, v_key = detect_bpm(v, SR), detect_key(v, SR)

    # ---- BEAT (with ffmpeg retry) ----
    path = os.path.join(BEATS_DIR, beat_name)
    b_raw = load_valid(path)
    if b_raw is None:
        st.error("Beat undecodable. Try a WAV beat or different MP3."); st.stop()
    b_bpm, b_key = detect_bpm(b_raw, SR), detect_key(b_raw, SR)

    # ---- SAFE MODE PROCESSING (NO ALIGNMENT) ----
    rate = (b_bpm / max(v_bpm, 1e-6))
    b = b_raw.copy()

    # time-stretch with fallback
    b_prev = b.copy()
    b = time_stretch_safe(b, rate)
    if len(b) < 1024 or np.max(np.abs(b)) == 0:
        b = b_prev  # revert

    # pitch-shift with fallback
    steps = 0
    if match_pitch:
        steps = semitone_distance(b_key, v_key)
        b_prev = b.copy()
        b = pitch_shift_safe(b, SR, steps)
        if len(b) < 1024 or np.max(np.abs(b)) == 0:
            b = b_prev  # revert

    b_aligned = b  # NO alignment in safe mode

    # length + level + guaranteed presence (+6 to +12 dB if needed)
    if len(b_aligned) < len(v): 
        b_aligned = np.pad(b_aligned, (0, len(v)-len(b_aligned)))
    else: 
        b_aligned = b_aligned[:len(v)]

    v_r = rms(v)
    b_r = rms(b_aligned)
    if b_r > 1e-6:
        b_aligned *= (v_r / b_r)
    boosts = 0
    while rms(b_aligned) < 0.5 * max(v_r, 1e-6) and boosts < 2:
        b_aligned *= 2.0
        boosts += 1

    mix = np.clip(0.55*v + 0.55*b_aligned, -1.0, 1.0).astype(np.float32)

    # ---- analyze remix + show ----
    remix_bpm, remix_key = detect_bpm(mix, SR), detect_key(mix, SR)
    st.subheader("Vocal (solo)"); st.audio(wav_bytes(v, SR))
    st.subheader("Beat (processed solo)"); st.audio(wav_bytes(b_aligned, SR))
    st.subheader("Mix"); mix_bytes = wav_bytes(mix, SR); st.audio(mix_bytes)

    report = {
        "vocal": {"bpm": round(v_bpm,2), "key": v_key},
        "beat_chosen": {"id": beat_name, "bpm_measured": round(b_bpm,2), "key_measured": b_key},
        "processing": {"stretch_rate": round(float(rate),4), "pitch_steps": int(steps)},
        "remix": {"bpm": round(remix_bpm,2), "key": remix_key},
        "levels_rms": {"vocal": round(v_r,5), "beat_processed": round(rms(b_aligned),5), "mix": round(rms(mix),5)},
        "beat_presence_ratio": round(float(rms(b_aligned)/(v_r+1e-9)), 3)
    }
    st.subheader("Proof (before vs after)")
    st.json(report)
    st.download_button("Download Remix WAV", data=mix_bytes, file_name=f"remix_{uuid.uuid4().hex}.wav", mime="audio/wav")
