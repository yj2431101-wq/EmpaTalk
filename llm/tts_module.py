"""Text-to-Speech module with two backends.

Primary:  edge-tts (Microsoft, online) — natural voices, requires internet.
Fallback: espeak-ng (offline)          — robotic but works without network.

Selects a voice based on listener profile (age, gender, timbre) from
Avamerg JSON, then synthesises audio and saves as a WAV file.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import edge_tts


# ── edge-tts voice table ────────────────────────────────────────────────────
# https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
_EDGE_VOICE_TABLE: dict[tuple[str, str], str] = {
    ("female", "high"):  "en-US-AvaNeural",
    ("female", "mid"):   "en-US-JennyNeural",
    ("female", "low"):   "en-US-AriaNeural",
    ("male",   "high"):  "en-US-RogerNeural",
    ("male",   "mid"):   "en-US-GuyNeural",
    ("male",   "low"):   "en-US-BrianNeural",
}
_EDGE_DEFAULT_VOICE = "en-US-JennyNeural"

_EDGE_RATE_BY_AGE: dict[str, str] = {
    "young": "+5%",
    "mid":   "+0%",
    "old":   "-5%",
}

# ── espeak-ng voice table (offline fallback) ─────────────────────────────────
# espeak-ng variant codes: en+f* = female, en+m* = male
_ESPEAK_VOICE_TABLE: dict[tuple[str, str], str] = {
    ("female", "high"):  "en+f3",
    ("female", "mid"):   "en+f2",
    ("female", "low"):   "en+f1",
    ("male",   "high"):  "en+m3",
    ("male",   "mid"):   "en+m2",
    ("male",   "low"):   "en+m1",
}
_ESPEAK_DEFAULT_VOICE = "en+f2"

_ESPEAK_SPEED_BY_AGE: dict[str, int] = {
    "young": 155,
    "mid":   145,
    "old":   130,
}


def _select_voice(listener_profile: Optional[dict]) -> dict:
    """Return voice params for both backends based on listener profile."""
    if not listener_profile:
        return {
            "edge_voice":   _EDGE_DEFAULT_VOICE,
            "edge_rate":    "+0%",
            "espeak_voice": _ESPEAK_DEFAULT_VOICE,
            "espeak_speed": 145,
        }

    gender = listener_profile.get("gender", "female").lower()
    timbre = listener_profile.get("timbre", "mid").lower()
    age    = listener_profile.get("age",    "young").lower()

    return {
        "edge_voice":   _EDGE_VOICE_TABLE.get((gender, timbre), _EDGE_DEFAULT_VOICE),
        "edge_rate":    _EDGE_RATE_BY_AGE.get(age, "+0%"),
        "espeak_voice": _ESPEAK_VOICE_TABLE.get((gender, timbre), _ESPEAK_DEFAULT_VOICE),
        "espeak_speed": _ESPEAK_SPEED_BY_AGE.get(age, 145),
    }


async def _synthesise_mp3(text: str, voice: str, rate: str, mp3_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(mp3_path)


def _mp3_to_wav(mp3_path: str, wav_path: str, sample_rate: int = 16000) -> None:
    """Convert MP3 → WAV (16 kHz mono) using ffmpeg if available, else pydub."""
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", str(sample_rate), "-ac", "1", wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(mp3_path).set_channels(1).set_frame_rate(sample_rate)
        audio.export(wav_path, format="wav")
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found and pydub not installed. "
            "Install ffmpeg: apt-get install ffmpeg"
        )


def _espeak_to_wav(text: str, voice: str, speed: int, wav_path: str, sample_rate: int) -> None:
    """Synthesise text with espeak-ng and resample to *sample_rate* Hz.

    Resampling is done with scipy.signal.resample_poly (no ffmpeg needed).
    Falls back to writing the raw espeak WAV if scipy is unavailable.
    """
    if not shutil.which("espeak-ng"):
        raise RuntimeError("espeak-ng not found. Install with: apt-get install espeak-ng")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        raw_wav = tmp.name

    try:
        subprocess.run(
            ["espeak-ng", "-v", voice, "-s", str(speed), text, "-w", raw_wav],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _resample_wav(raw_wav, wav_path, sample_rate)
    finally:
        if os.path.exists(raw_wav):
            os.remove(raw_wav)


def _resample_wav(src_path: str, dst_path: str, target_sr: int) -> None:
    """Read *src_path* WAV, resample to *target_sr* Hz, write *dst_path*."""
    import wave
    import struct
    import math

    try:
        from scipy.signal import resample_poly
        import numpy as np
        _resample_scipy(src_path, dst_path, target_sr, resample_poly, np)
        return
    except ImportError:
        pass

    # Pure-stdlib fallback: copy as-is (no resampling)
    import shutil as _shutil
    _shutil.copy2(src_path, dst_path)


def _resample_scipy(src_path: str, dst_path: str, target_sr: int, resample_poly, np) -> None:
    import wave, struct

    with wave.open(src_path, "rb") as wf:
        src_sr     = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        n_frames   = wf.getnframes()
        raw_bytes  = wf.readframes(n_frames)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    samples = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)

    # Convert to mono if stereo
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if src_sr != target_sr:
        from math import gcd
        g = gcd(src_sr, target_sr)
        samples = resample_poly(samples, target_sr // g, src_sr // g)

    out = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(dst_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(target_sr)
        wf.writeframes(out.tobytes())


class TTSModule:
    """Convert empathic response text to WAV audio.

    Tries edge-tts (online, natural quality) first; falls back to espeak-ng
    (offline, robotic) automatically on network errors.

    Args:
        listener_profile: Avamerg listener profile dict with age, gender,
                          timbre keys — used to pick an appropriate voice.
        sample_rate:      Output WAV sample rate (default 16 000 Hz, matching
                          EDTalk's mel-spectrogram pipeline).
        backend:          "auto" (try edge-tts, fall back to espeak),
                          "edge" (edge-tts only), or "espeak" (espeak only).
    """

    def __init__(
        self,
        listener_profile: Optional[dict] = None,
        sample_rate: int = 16000,
        backend: str = "auto",
    ):
        voices = _select_voice(listener_profile)
        self.edge_voice   = voices["edge_voice"]
        self.edge_rate    = voices["edge_rate"]
        self.espeak_voice = voices["espeak_voice"]
        self.espeak_speed = voices["espeak_speed"]
        self.sample_rate  = sample_rate
        self.backend      = backend

    def synthesise(self, text: str, output_path: str) -> str:
        """Synthesise *text* and write a WAV file to *output_path*.

        Args:
            text:        Empathic response sentence(s) to synthesise.
            output_path: Destination WAV file path.

        Returns:
            Absolute path to the written WAV file.
        """
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if self.backend == "espeak":
            _espeak_to_wav(text, self.espeak_voice, self.espeak_speed, output_path, self.sample_rate)
            return output_path

        # Try edge-tts (primary)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            mp3_path = tmp.name

        try:
            asyncio.run(_synthesise_mp3(text, self.edge_voice, self.edge_rate, mp3_path))
            _mp3_to_wav(mp3_path, output_path, self.sample_rate)
        except Exception as e:
            if self.backend == "edge":
                raise
            # Fallback to espeak-ng
            print(f"[TTSModule] edge-tts failed ({e}); falling back to espeak-ng.")
            _espeak_to_wav(text, self.espeak_voice, self.espeak_speed, output_path, self.sample_rate)
        finally:
            if os.path.exists(mp3_path):
                os.remove(mp3_path)

        return output_path

    def __repr__(self) -> str:
        return (
            f"TTSModule(edge_voice={self.edge_voice!r}, espeak_voice={self.espeak_voice!r}, "
            f"backend={self.backend!r}, sample_rate={self.sample_rate})"
        )
