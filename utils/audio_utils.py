from pathlib import Path
import json, numpy as np, soundfile as sf
import basic_pitch
from basic_pitch.inference import predict

def _sanity_check_wav(wav_path, min_seconds=2.0, min_rms=0.005):
    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    dur = len(y) / float(sr) if sr else 0.0
    rms = float(np.sqrt(np.mean(y**2))) if len(y) else 0.0
    if dur < min_seconds:
        raise ValueError(f"Recording too short ({dur:.2f}s). Please record ≥ {min_seconds}s.")
    if rms < min_rms:
        raise ValueError("Recording is too quiet/silent. Try a louder take.")

def _find_onnx_model() -> str:
    base = Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022"
    cands = list(base.rglob("*.onnx"))
    if not cands:
        raise FileNotFoundError(f"No ONNX model found under {base}.")
    return str(cands[0])

_ONNX_MODEL = _find_onnx_model()  # resolved once

def convert_audio_to_midi(wav_path: str, midi_out: str) -> str:
    """Transcribe WAV -> MIDI using Basic Pitch ONNX backend."""
    _sanity_check_wav(wav_path)
    _, midi_data, _ = predict(wav_path, model_or_model_path=_ONNX_MODEL)
    midi_data.write(midi_out)
    return midi_out
