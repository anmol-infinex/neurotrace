"""
NeuroTrace backend — EDF upload -> Keras model inference -> full prediction report.

Built from the exact training pipeline in python code.py:
  - Sample rate  : 128 Hz  (TARGET_FS)
  - Window size  : 10 s    (WINDOW_SIZE_SEC) → 1280 samples
  - Channels     : 22 TCP differential (TCP_CHANNELS)
  - Input shape  : (1280, 22)  — (samples, channels) = (SAMPLES_PER_WINDOW, NUM_CHANNELS)
  - Classes      : 0=Normal, 1=Pre-Seizure, 2=Seizure
  - Model file   : tusz_3class_eeg_model.h5
  - Annotation   : .csv_bi (primary) or .csv (fallback)
  - Sustained    : 2+ consecutive Seizure windows (CONSECUTIVE_SEIZURE_WINDOWS_REQUIRED)

Run locally:
    uvicorn app.main:app --reload --port 8000

Deploy: Render / Railway (see README.md). Vercel frontend calls this API.
"""

import re
import io
import csv
import json
import uuid
import shutil
import logging
from pathlib import Path
from datetime import timedelta
from typing import Optional, List, Tuple, Dict

import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurotrace")

# ============================================================
# CONFIG — extracted verbatim from python code.py constants
# ============================================================
MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "model" / "tusz_3class_eeg_model.h5"
)

# EEG pipeline — must match training exactly
TARGET_FS        = 128          # Hz  (TARGET_FS in training script)
WINDOW_SIZE_SEC  = 10           # seconds per window
SAMPLES_PER_WIN  = TARGET_FS * WINDOW_SIZE_SEC   # = 1280
NUM_CHANNELS     = 22           # 22-channel TUSZ TCP montage
STEP_SEC         = WINDOW_SIZE_SEC               # non-overlapping

# Seizure detection
CONSECUTIVE_SEIZURE_WINDOWS_REQUIRED = 2         # matches training script

# Classes — identical to LABEL_NAMES in training script
CLASS_NAMES = ["Normal", "Pre-Seizure", "Seizure"]

# Max upload size: 200 MB
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# 22-channel TUSZ TCP bipolar montage — exact order from training script
TCP_CHANNELS = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "A1-T3",  "T3-C3", "C3-CZ", "CZ-C4",
    "C4-T4",  "T4-A2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

# Electrode-level aliases (T7→T3, T8→T4, P7→T5, P8→T6)
CHANNEL_ALIASES: Dict[str, str] = {
    "T7": "T3", "T8": "T4",
    "P7": "T5", "P8": "T6",
}

# Bipolar channel pairs (same order as TCP_CHANNELS)
CHANNEL_PAIRS: List[Tuple[str, str]] = [
    ("FP1","F7"), ("F7","T3"), ("T3","T5"), ("T5","O1"),
    ("FP2","F8"), ("F8","T4"), ("T4","T6"), ("T6","O2"),
    ("A1","T3"),  ("T3","C3"), ("C3","CZ"), ("CZ","C4"),
    ("C4","T4"),  ("T4","A2"),
    ("FP1","F3"), ("F3","C3"), ("C3","P3"), ("P3","O1"),
    ("FP2","F4"), ("F4","C4"), ("C4","P4"), ("P4","O2"),
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="NeuroTrace Inference API",
    description=(
        "EDF upload → Keras 1D-CNN seizure model → full prediction report.\n"
        f"Model: tusz_3class_eeg_model.h5 | "
        f"Input: ({SAMPLES_PER_WIN}, {NUM_CHANNELS}) | {TARGET_FS} Hz | {WINDOW_SIZE_SEC}s windows"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://neurotrace-dashboard-eight.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Model file not found at {MODEL_PATH}. "
                    "Drop your tusz_3class_eeg_model.h5 there."
                ),
            )
        logger.info(f"Loading model from {MODEL_PATH}")
        _model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        logger.info(f"Model loaded. Input shape: {_model.input_shape}")
    return _model


# ============================================================
# EDF MAGIC-BYTE VALIDATION
# ============================================================
def validate_edf_magic(file_path: Path):
    """EDF files start with '0       ' (8 bytes)."""
    with open(file_path, "rb") as f:
        header = f.read(8)
    if not header.startswith(b"0       "):
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid EDF (magic bytes mismatch).",
        )


# ============================================================
# CHANNEL NAME CLEANING — mirrors clean_channel_name() in training script
# ============================================================
def clean_channel_name(name: str) -> str:
    s = str(name).strip().upper()
    s = re.sub(r"^EEG\s*", "", s)
    s = re.sub(r"_REF$", "", s)
    s = re.sub(r"-REF$", "", s)
    s = re.sub(r"_LE$", "", s)
    s = re.sub(r"-LE$", "", s)
    s = re.sub(r"_AR$", "", s)
    s = re.sub(r"-AR$", "", s)
    s = s.replace(" ", "")
    return CHANNEL_ALIASES.get(s, s)


# ============================================================
# BUILD 22-CHANNEL TCP MATRIX — mirrors build_tcp_channels()
# ============================================================
def build_tcp_channels(signals: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """
    Construct the 22-channel TUSZ TCP bipolar matrix from the signal dict.
    Priority: direct bipolar → derived from monopolar → zero-fill (missing).
    Returns (matrix shape=(22, n_samples), availability list).
    """
    result = []
    availability = []

    for (first, second) in CHANNEL_PAIRS:
        direct_name = f"{first}-{second}"
        reverse_name = f"{second}-{first}"

        direct_found = None
        for candidate in [direct_name, reverse_name]:
            if candidate in signals:
                direct_found = candidate
                break

        if direct_found is not None:
            data = signals[direct_found].astype(np.float32)
            if direct_found == reverse_name:
                data = -data
            result.append(data)
            availability.append("direct")
            continue

        # Derive from monopolar
        if first in signals and second in signals:
            a = signals[first].astype(np.float32)
            b = signals[second].astype(np.float32)
            min_len = min(len(a), len(b))
            result.append(a[:min_len] - b[:min_len])
            availability.append("derived")
            continue

        # Zero-fill missing
        lengths = [len(x) for x in signals.values() if len(x) > 0]
        length = max(lengths) if lengths else 0
        result.append(np.zeros(length, dtype=np.float32))
        availability.append("missing")

    if not result:
        raise ValueError("Could not construct any TCP channels.")

    min_length = min(len(x) for x in result)
    result = [x[:min_length] for x in result]
    return np.array(result, dtype=np.float32), availability


# ============================================================
# RESAMPLE — mirrors resample_window() in training script
# ============================================================
def resample_window(data: np.ndarray, raw_fs: float) -> np.ndarray:
    """Resample (22, raw_samples) window to (22, SAMPLES_PER_WIN)."""
    target_samples = SAMPLES_PER_WIN
    if data.shape[1] == target_samples:
        return data
    old_x = np.linspace(0, 1, data.shape[1])
    new_x = np.linspace(0, 1, target_samples)
    output = np.zeros((data.shape[0], target_samples), dtype=np.float32)
    for ch in range(data.shape[0]):
        output[ch] = np.interp(new_x, old_x, data[ch])
    return output


# ============================================================
# NORMALIZE — mirrors normalize_window() in training script
# ============================================================
def normalize_window(data: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalization."""
    mean = np.mean(data, axis=1, keepdims=True)
    std  = np.std(data,  axis=1, keepdims=True)
    std  = np.where(std < 1e-6, 1e-6, std)
    normalized = (data - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype(np.float32)


# ============================================================
# EDF → WINDOWED ARRAY — mirrors process_edf_into_windows()
# ============================================================
def load_and_window_edf(edf_path: Path):
    """
    Produces X shape (N, SAMPLES_PER_WIN, NUM_CHANNELS) = (N, 1280, 22).
    All preprocessing exactly matches the training script.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw_fs = float(raw.info["sfreq"])

    # Build monopolar signal dict
    signals: Dict[str, np.ndarray] = {}
    for name, data in zip(raw.ch_names, raw.get_data()):
        clean = clean_channel_name(name)
        if clean not in signals:
            signals[clean] = data.astype(np.float32)

    tcp_matrix, availability = build_tcp_channels(signals)

    any_missing = "missing" in availability
    channel_mismatch_warning = None
    if any_missing:
        missing_idx = [i for i, a in enumerate(availability) if a == "missing"]
        missing_names = [TCP_CHANNELS[i] for i in missing_idx]
        channel_mismatch_warning = (
            f"{len(missing_idx)} channel(s) not found and zero-filled: "
            f"{', '.join(missing_names)}. Predictions may be affected."
        )
        logger.warning(channel_mismatch_warning)

    window_samples = int(WINDOW_SIZE_SEC * raw_fs)
    if window_samples <= 0 or tcp_matrix.shape[1] < window_samples:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Recording too short ({tcp_matrix.shape[1] / raw_fs:.1f}s). "
                f"Need at least {WINDOW_SIZE_SEC}s."
            ),
        )

    windows = []
    starts_sec = []

    for start in range(0, tcp_matrix.shape[1] - window_samples + 1, window_samples):
        end = start + window_samples
        t_start = start / raw_fs
        window = tcp_matrix[:, start:end]             # (22, raw_samples)
        window = resample_window(window, raw_fs)      # (22, 1280)
        window = normalize_window(window)             # (22, 1280)
        window = window.T                             # (1280, 22)  ← .T matches training
        windows.append(window)
        starts_sec.append(t_start)

    if not windows:
        raise HTTPException(status_code=400, detail="No valid windows could be extracted.")

    X = np.array(windows, dtype=np.float32)           # (N, 1280, 22)

    # Keep raw (non-resampled, non-normalized) signal for preview
    preview_signal = tcp_matrix  # (22, n_samples_raw)

    return X, starts_sec, preview_signal, raw_fs, channel_mismatch_warning


# ============================================================
# INFERENCE
# ============================================================
def run_inference(X: np.ndarray) -> np.ndarray:
    """
    X: (N, 1280, 22) — already in (samples, channels) order from the T above.
    Model input_shape = (None, 1280, 22). No extra transpose needed.
    """
    model = get_model()
    probs = model.predict(X, batch_size=32, verbose=0)   # (N, 3)
    return probs


# ============================================================
# SUSTAINED SEIZURE DETECTION — mirrors training script logic
# ============================================================
def find_sustained_events(
    pred_classes: List[int],
    timestamps: List[float],
) -> List[dict]:
    events = []
    consecutive = 0
    event_start_sec = None

    for i, cls in enumerate(pred_classes):
        if cls == 2:  # Seizure
            if consecutive == 0:
                event_start_sec = timestamps[i]
            consecutive += 1
        else:
            if consecutive >= CONSECUTIVE_SEIZURE_WINDOWS_REQUIRED:
                end_sec = timestamps[i - 1] + WINDOW_SIZE_SEC
                events.append({
                    "start": _fmt(event_start_sec),
                    "end":   _fmt(end_sec),
                    "windows": consecutive,
                })
            consecutive = 0
            event_start_sec = None

    if consecutive >= CONSECUTIVE_SEIZURE_WINDOWS_REQUIRED:
        end_sec = timestamps[-1] + WINDOW_SIZE_SEC
        events.append({
            "start": _fmt(event_start_sec),
            "end":   _fmt(end_sec),
            "windows": consecutive,
        })
    return events


def _fmt(seconds: float) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ============================================================
# REPORT BUILDER — exact format from design.md §3
# ============================================================
def build_text_report(result: dict) -> str:
    lines = [
        "=" * 60,
        "TUSZ EEG USER EDF PREDICTION REPORT",
        "",
        "PREDICTION",
        "-" * 60,
        f"Final Prediction: {result['final_prediction']}",
        "",
        "AVERAGE PROBABILITIES",
        "-" * 60,
    ]
    for name, p in result["average_probabilities"].items():
        lines.append(f"{name}: {p:.4f}%")
    lines.append("-" * 60)
    for name, c in result["window_counts"].items():
        lines.append(f"{name}: {c}")
    lines.append(f"Total: {result['total_windows']}")
    lines.append("")
    lines.append("SUSTAINED SEIZURE EVENTS")
    lines.append("-" * 60)
    if result["sustained_events"]:
        for i, ev in enumerate(result["sustained_events"], 1):
            lines.append(
                f"Event {i}: {ev['start']} -> {ev['end']} | {ev['windows']} windows"
            )
    else:
        lines.append("None detected.")
    return "\n".join(lines)


# ============================================================
# ANNOTATION PARSER — mirrors parse_annotation_file() in training script
# ============================================================
def parse_tusz_annotations(
    annotation_path: Path,
    starts_sec: List[float],
) -> Optional[List[int]]:
    """
    Parse .csv_bi (primary) or .csv annotation files.
    Returns list of true class indices aligned to starts_sec,
    or None if the file cannot be parsed.

    Class mapping:
        "seiz" in line AND NOT "bckg"/"background" → 2 (Seizure)
        Otherwise → 0 (Normal / background)

    NOTE: Pre-Seizure (class 1) requires 15-minute look-ahead which is
    impractical per-window via annotation, so windows default to 0 or 2.
    """
    seizures = []
    try:
        with open(annotation_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lower = line.lower()
                if any(lower.startswith(p) for p in ("channel", "montage", "start")):
                    continue

                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 3:
                    continue

                numeric = []
                for p in parts:
                    try:
                        numeric.append(float(p))
                    except Exception:
                        pass

                if len(numeric) < 2:
                    continue

                start, end = numeric[0], numeric[1]
                if end <= start:
                    continue

                text = line.lower()
                is_seizure = "seiz" in text and "bckg" not in text and "background" not in text
                if is_seizure:
                    seizures.append((start, end))

    except Exception as e:
        logger.warning(f"Annotation parse warning: {e}")
        return None

    if not seizures:
        logger.warning("Annotation file has no seizure segments — treating as all-Normal.")

    # Deduplicate + sort
    seen = set()
    cleaned = []
    for seg in seizures:
        key = (round(seg[0], 4), round(seg[1], 4))
        if key not in seen:
            seen.add(key)
            cleaned.append(seg)
    cleaned.sort(key=lambda x: x[0])

    # Assign label to each window (majority-overlap with any seizure segment)
    true_classes = []
    for w_start in starts_sec:
        w_end = w_start + WINDOW_SIZE_SEC
        label = 0
        for sz_start, sz_end in cleaned:
            overlap = max(0.0, min(w_end, sz_end) - max(w_start, sz_start))
            if overlap >= 5.0:   # ≥5 s overlap → seizure (matches training)
                label = 2
                break
        true_classes.append(label)

    return true_classes


# ============================================================
# ROC + CONFUSION MATRIX
# ============================================================
def build_roc_and_confusion(
    probs: np.ndarray,
    pred_classes: List[int],
    true_classes: Optional[List[int]],
    out_dir: Path,
):
    if true_classes is None:
        return None, None, None

    n_classes = len(CLASS_NAMES)
    y_true = np.array(true_classes)
    y_pred = np.array(pred_classes)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # ROC
    plt.figure(figsize=(8, 7))
    for i, name in enumerate(CLASS_NAMES):
        pos = int(np.sum(y_true_bin[:, i] == 1))
        neg = int(np.sum(y_true_bin[:, i] == 0))
        if pos == 0 or neg == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("TUSZ EEG ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_path = out_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    threshold = cm.max() / 2 if cm.size else 0
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("TUSZ EEG Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(CLASS_NAMES, rotation=20)
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(n_classes):
        for j in range(n_classes):
            v = cm[i, j]
            ax.text(
                j, i, str(v), ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if v > threshold else "black",
            )
    cm_path = out_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    cm_text_path = out_dir / "confusion_matrix.txt"
    with open(cm_text_path, "w") as f:
        f.write("Confusion Matrix (rows=true, cols=predicted)\n")
        f.write("Classes: " + ", ".join(CLASS_NAMES) + "\n\n")
        f.write(np.array2string(cm))

    return roc_path, cm_path, cm_text_path


# ============================================================
# ROUTES
# ============================================================
@app.get("/health")
def health():
    model_exists = MODEL_PATH.exists()
    info = {
        "status": "ok",
        "model_loaded": model_exists,
        "model_path": str(MODEL_PATH),
        "config": {
            "sample_rate_hz": TARGET_FS,
            "window_sec": WINDOW_SIZE_SEC,
            "samples_per_window": SAMPLES_PER_WIN,
            "num_channels": NUM_CHANNELS,
            "classes": CLASS_NAMES,
            "input_shape": [SAMPLES_PER_WIN, NUM_CHANNELS],
        },
    }
    return info


@app.post("/predict")
async def predict(
    edf_file: UploadFile = File(...),
    annotation_file: Optional[UploadFile] = File(None),
):
    # --- Validation ---
    if not edf_file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Please upload a .edf file.")

    content = await edf_file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)/1024/1024:.1f} MB). Max: {MAX_UPLOAD_BYTES//1024//1024} MB.",
        )

    job_id = str(uuid.uuid4())
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    edf_path = job_dir / edf_file.filename
    with open(edf_path, "wb") as f:
        f.write(content)

    validate_edf_magic(edf_path)

    # --- EDF → windows ---
    X, starts_sec, preview_signal, raw_fs, channel_mismatch_warning = (
        load_and_window_edf(edf_path)
    )

    # --- Inference ---
    probs = run_inference(X)                         # (N, 3)
    pred_classes = np.argmax(probs, axis=1).tolist() # [int, ...]

    # --- Aggregation ---
    window_counts = {
        name: int(np.sum(np.array(pred_classes) == i))
        for i, name in enumerate(CLASS_NAMES)
    }
    avg_probs = {
        name: float(np.mean(probs[:, i]) * 100)
        for i, name in enumerate(CLASS_NAMES)
    }
    final_prediction = CLASS_NAMES[int(np.argmax(list(avg_probs.values())))]

    sustained_events = find_sustained_events(pred_classes, starts_sec)

    # --- Optional annotation → ROC / confusion ---
    true_classes = None
    if annotation_file is not None:
        ann_path = job_dir / annotation_file.filename
        ann_content = await annotation_file.read()
        with open(ann_path, "wb") as f:
            f.write(ann_content)
        true_classes = parse_tusz_annotations(ann_path, starts_sec)

    roc_path, cm_path, cm_text_path = build_roc_and_confusion(
        probs, pred_classes, true_classes, job_dir
    )

    # --- Signal preview (downsample to ~50 Hz for the frontend graph) ---
    step = max(1, int(raw_fs // 50))
    preview_data = preview_signal[:, ::step].tolist()  # (22, preview_samples)

    result = {
        "job_id": job_id,
        "final_prediction": final_prediction,
        "average_probabilities": avg_probs,
        "window_counts": window_counts,
        "total_windows": int(len(pred_classes)),
        "sustained_events": sustained_events,
        "channels_used": TCP_CHANNELS,
        "channel_mismatch_warning": channel_mismatch_warning,
        "window_predictions": [
            {
                "start_sec": s,
                "class": CLASS_NAMES[c],
                "probabilities": probs[i].tolist(),
            }
            for i, (s, c) in enumerate(zip(starts_sec, pred_classes))
        ],
        "signal_preview": {
            "channels": TCP_CHANNELS,
            "sample_rate_used_for_preview": int(raw_fs // step),
            "data": preview_data,
        },
        "has_ground_truth": true_classes is not None,
    }

    report_text = build_text_report(result)
    (job_dir / "report.txt").write_text(report_text, encoding="utf-8")
    (job_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["report_text"] = report_text
    result["download_urls"] = {
        "report_txt":    f"/predict/{job_id}/report",
        "result_json":   f"/predict/{job_id}/json",
        "roc_png":       f"/predict/{job_id}/roc.png"       if roc_path    else None,
        "confusion_png": f"/predict/{job_id}/confusion.png" if cm_path     else None,
        "confusion_txt": f"/predict/{job_id}/confusion.txt" if cm_text_path else None,
    }

    logger.info(
        f"Job {job_id}: {final_prediction}, {len(pred_classes)} windows, "
        f"{len(sustained_events)} sustained events"
    )
    return JSONResponse(result)


@app.get("/predict/{job_id}/report")
def get_report(job_id: str):
    path = OUTPUT_DIR / job_id / "report.txt"
    if not path.exists():
        raise HTTPException(404, "Job not found.")
    return FileResponse(path, filename="prediction_report.txt", media_type="text/plain")


@app.get("/predict/{job_id}/json")
def get_json(job_id: str):
    path = OUTPUT_DIR / job_id / "result.json"
    if not path.exists():
        raise HTTPException(404, "Job not found.")
    return FileResponse(path, filename="result.json", media_type="application/json")


@app.get("/predict/{job_id}/roc.png")
def get_roc(job_id: str):
    path = OUTPUT_DIR / job_id / "roc_curve.png"
    if not path.exists():
        raise HTTPException(
            404,
            "ROC not available. Supply a ground-truth annotation file (.csv_bi/.csv)."
        )
    return FileResponse(path, media_type="image/png")


@app.get("/predict/{job_id}/confusion.png")
def get_confusion_png(job_id: str):
    path = OUTPUT_DIR / job_id / "confusion_matrix.png"
    if not path.exists():
        raise HTTPException(
            404,
            "Confusion matrix not available. Supply a ground-truth annotation file."
        )
    return FileResponse(path, media_type="image/png")


@app.get("/predict/{job_id}/confusion.txt")
def get_confusion_txt(job_id: str):
    path = OUTPUT_DIR / job_id / "confusion_matrix.txt"
    if not path.exists():
        raise HTTPException(
            404,
            "Confusion matrix not available. Supply a ground-truth annotation file."
        )
    return FileResponse(path, media_type="text/plain")
