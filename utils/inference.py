from pathlib import Path
import json
import numpy as np
import tensorflow as tf

# ----- Paths -----
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH  = ROOT / "model" / "best_cnn.keras"
LABELS_PATH = Path(__file__).resolve().with_name("label_map.json")

# ----- Load model + labels -----
MODEL = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
COMPOSERS = json.load(open(LABELS_PATH))   # e.g. ["Bach","Beethoven","Chopin","Mozart"]

SEQ_T  = 512
N_KEYS = 88

def _prep_roll(pr: np.ndarray) -> np.ndarray:
    pr = np.asarray(pr)
    if pr.ndim != 2:
        raise ValueError(f"Expected 2D piano-roll, got {pr.shape}")

    # Accept (88,T) and flip to (T,88)
    if pr.shape[0] == N_KEYS and pr.shape[1] != N_KEYS:
        pr = pr.T

    # MATCH TRAINING: binary {0,1}
    pr = (pr > 0).astype(np.uint8)

    # Left-aligned pad/trim to 512 frames
    T = pr.shape[0]
    if T < SEQ_T:
        pr = np.pad(pr, ((0, SEQ_T - T), (0, 0)), mode="constant")
    elif T > SEQ_T:
        pr = pr[:SEQ_T, :]

    x = pr.reshape(1, SEQ_T, N_KEYS, 1)    # (1,512,88,1)
    return x

def predict_composer(piano_roll: np.ndarray):
    """
    Returns (probabilities_dict, processed_roll_for_viz)
    """
    x = _prep_roll(piano_roll)                 # (1,512,88,1), values 0..127
    probs = MODEL.predict(x, verbose=0)[0]     # (4,)

    # Map in index order *without* sorting labels; they already match training.
    order = np.argsort(probs)[::-1]
    probs_dict = { COMPOSERS[i]: float(probs[i]) for i in order }

    # return a (88,512) roll for your plotter
    viz_roll = x[0, :, :, 0].T                 # (88,512)
    return probs_dict, viz_roll
