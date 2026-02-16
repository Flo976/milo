import base64
import io
import logging
import struct

import numpy as np
import soundfile as sf

from app.config import settings

logger = logging.getLogger("milo")


class AudioValidationError(Exception):
    pass


def decode_base64_audio(data: str) -> tuple[np.ndarray, int]:
    """Decode base64-encoded audio to numpy array + sample rate."""
    try:
        raw = base64.b64decode(data)
    except Exception as e:
        raise AudioValidationError(f"Invalid base64: {e}")

    if len(raw) > settings.upload_max_bytes:
        raise AudioValidationError(
            f"Audio exceeds {settings.upload_max_bytes // (1024*1024)} MB limit"
        )

    buf = io.BytesIO(raw)
    try:
        audio, sr = sf.read(buf, dtype="float32")
    except Exception as e:
        raise AudioValidationError(f"Cannot read audio: {e}")

    return audio, sr


def ensure_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Convert audio to mono 16kHz float32."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except ImportError:
            raise AudioValidationError("librosa required for resampling")

    return audio.astype(np.float32)


def validate_duration(audio: np.ndarray, sr: int = 16000) -> float:
    """Check audio is within STT duration limit. Returns duration in seconds."""
    duration = len(audio) / sr
    if duration > settings.stt_max_duration_s:
        raise AudioValidationError(
            f"Audio duration {duration:.1f}s exceeds {settings.stt_max_duration_s}s limit"
        )
    return duration


def audio_to_base64_wav(audio: np.ndarray, sr: int = 16000) -> str:
    """Encode numpy audio array to base64 WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def pcm_bytes_to_float32(data: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Convert raw PCM 16-bit LE bytes to float32 numpy array."""
    n_samples = len(data) // 2
    samples = struct.unpack(f"<{n_samples}h", data[:n_samples * 2])
    return np.array(samples, dtype=np.float32) / 32768.0
