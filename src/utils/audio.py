from pathlib import Path

import numpy as np


def save_audio(audio: np.ndarray, path: Path, sample_rate: int = 22050) -> None:
    import soundfile as sf
    sf.write(str(path), audio, sample_rate)


def load_audio(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(str(path))

    if sample_rate and sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    return audio.astype(np.float32), sr


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 10 ** (target_db / 20)
        audio = audio * (target_rms / rms)
    return np.clip(audio, -1.0, 1.0)
