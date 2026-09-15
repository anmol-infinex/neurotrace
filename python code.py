
# ============================================================
# TUH EEG SEIZURE CORPUS (TUSZ) v2.0.3
# ============================================================
#
# EEG SEIZURE DETECTION + PRE-SEIZURE PREDICTION
#
# TensorFlow / Keras 1D CNN Deep Learning Pipeline
#
# DATASET:
#   Temple University Hospital Seizure Corpus
#   TUSZ v2.0.3
#
# DATASET STRUCTURE:
#
# v2.0.3/
# └── edf/
#     ├── train/
#     │   └── patient/session/montage/
#     │       ├── file.edf
#     │       ├── file.csv
#     │       └── file.csv_bi
#     │
#     ├── dev/
#     │   └── patient/session/montage/
#     │       ├── file.edf
#     │       ├── file.csv
#     │       └── file.csv_bi
#     │
#     └── eval/
#         └── patient/session/montage/
#             ├── file.edf
#             ├── file.csv
#             └── file.csv_bi
#
#
# CLASSES:
#   0 = NORMAL / BACKGROUND
#   1 = PRE-SEIZURE / PREICTAL
#   2 = SEIZURE / ICTAL
#
#
# PRE-SEIZURE:
#   15 minutes before seizure onset
#
#
# EEG:
#   Sampling frequency = 128 Hz
#   Window = 10 seconds
#   Samples = 1280
#   Channels = 22 TCP differential derivations
#
#
# AUTOMATIC OUTPUTS:
#
#   eeg_analysis_results/
#
#   ├── trained_tusz_model_architecture.png
#   ├── training_accuracy.png
#   ├── training_loss.png
#   ├── dev_confusion_matrix.png
#   ├── dev_roc_curve.png
#   ├── dev_classification_report.txt
#   ├── dev_evaluation_metrics.json
#   ├── eval_confusion_matrix.png
#   ├── eval_roc_curve.png
#   ├── eval_classification_report.txt
#   ├── eval_evaluation_metrics.json
#   └── user_edf_prediction_report.txt
#
#
# USER MODE:
#
# After training, the program allows the user to select
# ANY compatible EDF file from the computer.
#
# The user does NOT need to provide:
#   .csv
#   .csv_bi
#
# The model performs prediction directly from EDF.
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import re
import csv
import json
import glob
import warnings
from pathlib import Path


# ============================================================
# AUTO-DETECT VIRTUAL ENVIRONMENT WITH TENSORFLOW
# ============================================================

_venv_python = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".venv",
    "Scripts",
    "python.exe"
)

if os.path.exists(_venv_python) and sys.executable.lower() != _venv_python.lower():
    try:
        import keras
    except ImportError:
        try:
            import tensorflow
        except ImportError:
            import subprocess
            sys.exit(
                subprocess.call([_venv_python] + sys.argv)
            )


# ============================================================
# REAL-TIME OUTPUT
# ============================================================

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

warnings.filterwarnings("ignore")


# ============================================================
# NUMPY
# ============================================================

import numpy as np


# ============================================================
# MATPLOTLIB
# ============================================================

try:

    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True

except ImportError:

    MATPLOTLIB_AVAILABLE = False

    print(
        "WARNING: matplotlib is not installed."
    )

    print(
        "Install using: pip install matplotlib"
    )


# ============================================================
# SCIKIT-LEARN
# ============================================================

try:

    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_curve,
        auc,
        roc_auc_score
    )

    from sklearn.preprocessing import label_binarize

    from sklearn.utils.class_weight import (
        compute_class_weight
    )

    SKLEARN_AVAILABLE = True

except ImportError:

    SKLEARN_AVAILABLE = False

    print(
        "WARNING: scikit-learn is not installed."
    )

    print(
        "Install using: pip install scikit-learn"
    )


# ============================================================
# ============================================================
# DEEP LEARNING BACKEND (KERAS 3 / PYTORCH CUDA / TENSORFLOW)
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

GPU_NAME = None
ACTIVE_BACKEND = "CPU"

# Automatically enable PyTorch GPU acceleration on Windows if CUDA is available
if "KERAS_BACKEND" not in os.environ:
    try:
        import torch
        if torch.cuda.is_available():
            os.environ["KERAS_BACKEND"] = "torch"
            GPU_NAME = torch.cuda.get_device_name(0)
            ACTIVE_BACKEND = f"Keras 3 + PyTorch CUDA GPU ({GPU_NAME})"
    except ImportError:
        pass

try:

    import keras

    from keras import (
        layers,
        models,
        callbacks
    )

    if not GPU_NAME:
        ACTIVE_BACKEND = f"Keras 3 ({keras.config.backend().upper()} Backend)"

    TF_AVAILABLE = True

except ImportError:

    try:

        import tensorflow as tf

        from tensorflow import keras

        from tensorflow.keras import (
            layers,
            models,
            callbacks
        )

        ACTIVE_BACKEND = "TensorFlow (CPU)"
        TF_AVAILABLE = True

    except ImportError:

        TF_AVAILABLE = False

        print(
            "\n=================================================="
        )

        print(
            "WARNING: Deep learning framework not installed."
        )

        print(
            "Install using:"
        )

        print(
            "pip install keras torch"
        )

        print(
            "=================================================="
        )


# ============================================================
# MNE
# ============================================================

try:

    import mne

    mne.set_log_level(
        "ERROR"
    )

    MNE_AVAILABLE = True

except ImportError:

    MNE_AVAILABLE = False

    print(
        "WARNING: MNE is not installed."
    )

    print(
        "Install using:"
    )

    print(
        "pip install mne"
    )


# ============================================================
# TKINTER
# ============================================================

try:

    import tkinter as tk

    from tkinter import (
        filedialog,
        messagebox
    )

    TKINTER_AVAILABLE = True

except ImportError:

    TKINTER_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
#
# Point this to:
#
# c:\...\TUSZ\v2.0.3
#
# NOT directly to train/dev/eval.
#
# ============================================================

DATASET_ROOT = str(
    Path(__file__).resolve().parent
)


# ============================================================
# TUSZ SPLITS
# ============================================================

EDF_ROOT = os.path.join(
    DATASET_ROOT,
    "edf"
)

TRAIN_ROOT = os.path.join(
    EDF_ROOT,
    "train"
)

DEV_ROOT = os.path.join(
    EDF_ROOT,
    "dev"
)

EVAL_ROOT = os.path.join(
    EDF_ROOT,
    "eval"
)

 
# ============================================================
# MODEL
# ============================================================

MODEL_SAVE_PATH = os.path.join(
    DATASET_ROOT,
    "tusz_3class_eeg_model.h5"
)


# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR = os.path.join(
    DATASET_ROOT,
    "tusz_eeg_analysis_results987"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_IMAGE_PATH = os.path.join(
    RESULTS_DIR,
    "trained_tusz_model_architecture.png"
)

TRAINING_ACC_PATH = os.path.join(
    RESULTS_DIR,
    "training_accuracy.png"
)

TRAINING_LOSS_PATH = os.path.join(
    RESULTS_DIR,
    "training_loss.png"
)

DEV_CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "dev_confusion_matrix.png"
)

DEV_ROC_CURVE_PATH = os.path.join(
    RESULTS_DIR,
    "dev_roc_curve.png"
)

DEV_CLASSIFICATION_REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "dev_classification_report.txt"
)

DEV_METRICS_JSON_PATH = os.path.join(
    RESULTS_DIR,
    "dev_evaluation_metrics.json"
)

EVAL_CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "eval_confusion_matrix.png"
)

EVAL_ROC_CURVE_PATH = os.path.join(
    RESULTS_DIR,
    "eval_roc_curve.png"
)

EVAL_CLASSIFICATION_REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "eval_classification_report.txt"
)

EVAL_METRICS_JSON_PATH = os.path.join(
    RESULTS_DIR,
    "eval_evaluation_metrics.json"
)

USER_EDF_REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "user_edf_prediction_report.txt"
)


# ============================================================
# EEG CONFIGURATION
# ============================================================

WINDOW_SIZE_SEC = 10

PREDICTION_TIME_SEC = 1800
# 15 minutes

TARGET_FS = 128

SAMPLES_PER_WINDOW = (
    WINDOW_SIZE_SEC *
    TARGET_FS
)

NUM_CLASSES = 3


# ============================================================
# LABELS
# ============================================================

LABEL_NAMES = {

    0: "Normal",

    1: "Pre-Seizure",

    2: "Seizure"
}


# ============================================================
# SEIZURE DETECTION CONFIGURATION
# ============================================================

CONSECUTIVE_SEIZURE_WINDOWS_REQUIRED = 2


# ============================================================
# TUSZ TCP 22 DIFFERENTIAL CHANNELS
# ============================================================

TCP_CHANNELS = [

    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",

    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",

    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",

    "C4-T4",
    "T4-A2",

    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",

    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2"
]

NUM_CHANNELS = len(
    TCP_CHANNELS
)


# ============================================================
# MONOPOLAR ELECTRODE CHANNELS
# ============================================================

ELECTRODE_NAMES = [

    "FP1",
    "F7",
    "T3",
    "T5",
    "O1",

    "FP2",
    "F8",
    "T4",
    "T6",
    "O2",

    "A1",
    "C3",
    "CZ",
    "C4",
    "A2",

    "F3",
    "P3",
    "F4",
    "P4"
]


# ============================================================
# CHANNEL ALIASES
# ============================================================

CHANNEL_ALIASES = {

    "T7": "T3",
    "T8": "T4",

    "P7": "T5",
    "P8": "T6"
}


# ============================================================
# CHANNEL PAIRS
# ============================================================

CHANNEL_PAIRS = [

    ("FP1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),

    ("FP2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),

    ("A1", "T3"),
    ("T3", "C3"),
    ("C3", "CZ"),
    ("CZ", "C4"),

    ("C4", "T4"),
    ("T4", "A2"),

    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),

    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2")
]


# ============================================================
# UTILITY
# ============================================================

def format_time(seconds):

    seconds = max(
        0,
        int(seconds)
    )

    hours = (
        seconds // 3600
    )

    minutes = (
        seconds % 3600
    ) // 60

    secs = (
        seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# CLEAN CHANNEL NAME
# ============================================================

def clean_channel_name(name):

    s = str(
        name
    ).strip().upper()

    s = re.sub(
        r"^EEG\s*",
        "",
        s
    )

    s = re.sub(
        r"_REF$",
        "",
        s
    )

    s = re.sub(
        r"-REF$",
        "",
        s
    )

    s = re.sub(
        r"_LE$",
        "",
        s
    )

    s = re.sub(
        r"-LE$",
        "",
        s
    )

    s = re.sub(
        r"_AR$",
        "",
        s
    )

    s = re.sub(
        r"-AR$",
        "",
        s
    )

    s = s.replace(
        " ",
        ""
    )

    s = CHANNEL_ALIASES.get(
        s,
        s
    )

    return s


# ============================================================
# CLEAN BIPOLAR CHANNEL
# ============================================================

def clean_bipolar_name(name):

    s = str(
        name
    ).strip().upper()

    s = s.replace(
        "EEG ",
        ""
    )

    s = s.replace(
        "_REF",
        ""
    )

    s = s.replace(
        "-REF",
        ""
    )

    s = s.replace(
        " ",
        ""
    )

    parts = re.split(
        r"[-/]",
        s
    )

    if len(parts) != 2:

        return None

    first = CHANNEL_ALIASES.get(
        parts[0],
        parts[0]
    )

    second = CHANNEL_ALIASES.get(
        parts[1],
        parts[1]
    )

    return (
        first,
        second
    )


# ============================================================
# EDF READER
# ============================================================

def read_edf_file(edf_path):

    """
    Read EDF using MNE.

    MNE is preferred for TUSZ because TUSZ contains
    multiple montage/reference configurations.
    """

    if not MNE_AVAILABLE:

        raise RuntimeError(
            "MNE is required for TUSZ EDF reading.\n"
            "Install using: pip install mne"
        )

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False
    )

    signals = {}

    for name, data in zip(
        raw.ch_names,
        raw.get_data()
    ):

        clean_name = clean_channel_name(
            name
        )

        if clean_name not in signals:

            signals[
                clean_name
            ] = np.asarray(
                data,
                dtype=np.float32
            )

    fs = float(
        raw.info["sfreq"]
    )

    duration = float(
        raw.times[-1]
        if len(raw.times) > 0
        else 0
    )

    return (
        signals,
        fs,
        duration,
        raw.ch_names
    )


# ============================================================
# BUILD TCP 22 CHANNELS
# ============================================================

def build_tcp_channels(
    signals
):

    """
    Create the exact 22 TUSZ TCP differential channels.

    If direct bipolar channels are available,
    they are used.

    Otherwise bipolar channels are calculated
    from monopolar electrode signals.

    Missing channels are zero-filled only when
    absolutely necessary.
    """

    result = []

    availability = []

    for first, second in CHANNEL_PAIRS:

        direct_name = (
            f"{first}-{second}"
        )

        reverse_name = (
            f"{second}-{first}"
        )

        # ----------------------------------------------------
        # Direct bipolar
        # ----------------------------------------------------

        direct_candidates = [

            direct_name,

            reverse_name
        ]

        direct_found = None

        for candidate in direct_candidates:

            if candidate in signals:

                direct_found = candidate

                break

        if direct_found is not None:

            data = np.asarray(
                signals[
                    direct_found
                ],
                dtype=np.float32
            )

            if direct_found == reverse_name:

                data = -data

            result.append(
                data
            )

            availability.append(
                "direct"
            )

            continue

        # ----------------------------------------------------
        # Derive from monopolar
        # ----------------------------------------------------

        if (
            first in signals
            and
            second in signals
        ):

            a = np.asarray(
                signals[first],
                dtype=np.float32
            )

            b = np.asarray(
                signals[second],
                dtype=np.float32
            )

            min_len = min(
                len(a),
                len(b)
            )

            data = (
                a[:min_len] -
                b[:min_len]
            )

            result.append(
                data
            )

            availability.append(
                "derived"
            )

            continue

        # ----------------------------------------------------
        # Missing channel
        # ----------------------------------------------------

        lengths = [
            len(x)
            for x in signals.values()
            if len(x) > 0
        ]

        length = (
            max(lengths)
            if lengths
            else 0
        )

        result.append(
            np.zeros(
                length,
                dtype=np.float32
            )
        )

        availability.append(
            "missing"
        )

    # --------------------------------------------------------
    # Equalize length
    # --------------------------------------------------------

    if not result:

        raise ValueError(
            "Could not construct TCP channels."
        )

    min_length = min(
        len(x)
        for x in result
    )

    result = [

        x[:min_length]

        for x in result
    ]

    matrix = np.asarray(
        result,
        dtype=np.float32
    )

    return (
        matrix,
        availability
    )


# ============================================================
# PARSE TUSZ ANNOTATION FILE
# ============================================================

def parse_annotation_file(
    annotation_path
):

    """
    Parse TUSZ .csv_bi or .csv annotation.

    We primarily use .csv_bi because it provides
    background/seizure term annotations.

    The parser is intentionally tolerant because
    annotation headers may vary between releases.
    """

    seizures = []

    if not annotation_path:

        return seizures

    if not os.path.exists(
        annotation_path
    ):

        return seizures

    try:

        with open(
            annotation_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:

                    continue

                if line.startswith(
                    "#"
                ):

                    continue

                lower = line.lower()

                if (
                    lower.startswith(
                        "channel"
                    )
                    or
                    lower.startswith(
                        "montage"
                    )
                    or
                    lower.startswith(
                        "start"
                    )
                ):

                    continue

                parts = [
                    x.strip()
                    for x in line.split(",")
                ]

                if len(parts) < 3:

                    continue

                # ------------------------------------------------
                # Find numeric start/end pair.
                # ------------------------------------------------

                numeric = []

                for p in parts:

                    try:

                        numeric.append(
                            float(p)
                        )

                    except Exception:

                        pass

                if len(numeric) < 2:

                    continue

                # Most TUSZ annotation lines have
                # start/stop time values.
                #
                # Use the first valid pair.

                start = numeric[0]

                end = numeric[1]

                if end <= start:

                    continue

                # ------------------------------------------------
                # Find seizure term.
                # ------------------------------------------------

                text = (
                    line.lower()
                )

                is_seizure = (

                    "seiz" in text

                    and

                    "bckg" not in text

                    and

                    "background" not in text
                )

                if is_seizure:

                    seizures.append(
                        (
                            start,
                            end,
                            "seiz"
                        )
                    )

    except Exception as e:

        print(
            f"Annotation parsing warning: "
            f"{e}"
        )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    cleaned = []

    seen = set()

    for item in seizures:

        key = (
            round(item[0], 4),
            round(item[1], 4)
        )

        if key not in seen:

            seen.add(key)

            cleaned.append(
                item
            )

    cleaned.sort(
        key=lambda x: x[0]
    )

    return cleaned


# ============================================================
# FIND ANNOTATION
# ============================================================

def find_annotation_file(
    edf_path
):

    base = os.path.splitext(
        edf_path
    )[0]

    csv_bi = (
        base +
        ".csv_bi"
    )

    csv_file = (
        base +
        ".csv"
    )

    if os.path.exists(
        csv_bi
    ):

        return csv_bi

    if os.path.exists(
        csv_file
    ):

        return csv_file

    return None


# ============================================================
# LABEL WINDOW
# ============================================================

def calculate_window_label(
    t_start,
    t_end,
    seizures
):

    # --------------------------------------------------------
    # Seizure
    #
    # At least 5 seconds of seizure overlap.
    # --------------------------------------------------------

    for (
        sz_start,
        sz_end,
        _
    ) in seizures:

        overlap_start = max(
            t_start,
            sz_start
        )

        overlap_end = min(
            t_end,
            sz_end
        )

        overlap = (
            overlap_end -
            overlap_start
        )

        if overlap >= 5.0:

            return 2

    # --------------------------------------------------------
    # Pre-seizure
    # --------------------------------------------------------

    for (
        sz_start,
        sz_end,
        _
    ) in seizures:

        preictal_start = max(
            0,
            sz_start -
            PREDICTION_TIME_SEC
        )

        if (
            preictal_start
            <=
            t_start
            <
            sz_start
        ):

            return 1

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------

    return 0


# ============================================================
# RESAMPLE
# ============================================================

def resample_window(
    data,
    raw_fs,
    target_fs
):

    target_samples = int(
        WINDOW_SIZE_SEC *
        target_fs
    )

    if data.shape[1] == target_samples:

        return data

    old_x = np.linspace(
        0,
        1,
        data.shape[1]
    )

    new_x = np.linspace(
        0,
        1,
        target_samples
    )

    output = np.zeros(
        (
            data.shape[0],
            target_samples
        ),
        dtype=np.float32
    )

    for channel in range(
        data.shape[0]
    ):

        output[channel] = np.interp(
            new_x,
            old_x,
            data[channel]
        )

    return output


# ============================================================
# NORMALIZE EEG
# ============================================================

def normalize_window(
    data
):

    mean = np.mean(
        data,
        axis=1,
        keepdims=True
    )

    std = np.std(
        data,
        axis=1,
        keepdims=True
    )

    std = np.where(
        std < 1e-6,
        1e-6,
        std
    )

    normalized = (
        data - mean
    ) / std

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# PROCESS EDF
# ============================================================

def process_edf_into_windows(
    edf_path,
    seizures=None,
    create_labels=True
):

    """
    Convert EDF into:
        (N, 1280, 22)

    For user prediction:
        create_labels=False
        annotations are NOT required.
    """

    if seizures is None:

        seizures = []

    print(
        f"\nReading EDF:"
        f"\n{edf_path}",
        flush=True
    )

    signals, raw_fs, duration, raw_names = (
        read_edf_file(
            edf_path
        )
    )

    print(
        f"Original sampling frequency: "
        f"{raw_fs:.2f} Hz"
    )

    print(
        f"Recording duration: "
        f"{duration:.2f} seconds"
    )

    tcp_matrix, availability = (
        build_tcp_channels(
            signals
        )
    )

    # --------------------------------------------------------
    # Window calculation
    # --------------------------------------------------------

    window_samples = int(
        WINDOW_SIZE_SEC *
        raw_fs
    )

    if (
        window_samples <= 0
        or
        tcp_matrix.shape[1]
        <
        window_samples
    ):

        return (
            np.empty(
                (
                    0,
                    SAMPLES_PER_WINDOW,
                    NUM_CHANNELS
                ),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=np.int32
            ),
            []
        )

    windows = []

    labels = []

    timestamps = []

    # --------------------------------------------------------
    # Non-overlapping 10-second windows
    # --------------------------------------------------------

    for start in range(
        0,
        tcp_matrix.shape[1] -
        window_samples +
        1,
        window_samples
    ):

        end = (
            start +
            window_samples
        )

        t_start = (
            start /
            raw_fs
        )

        t_end = (
            end /
            raw_fs
        )

        window = tcp_matrix[
            :,
            start:end
        ]

        # ----------------------------------------------------
        # Resample
        # ----------------------------------------------------

        window = resample_window(
            window,
            raw_fs,
            TARGET_FS
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        window = normalize_window(
            window
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        if create_labels:

            label = calculate_window_label(
                t_start,
                t_end,
                seizures
            )

            labels.append(
                label
            )

        # ----------------------------------------------------
        # TensorFlow format
        # ----------------------------------------------------

        window = window.T

        windows.append(
            window
        )

        timestamps.append(
            (
                t_start,
                t_end
            )
        )

    X = np.asarray(
        windows,
        dtype=np.float32
    )

    if create_labels:

        y = np.asarray(
            labels,
            dtype=np.int32
        )

    else:

        y = np.full(
            len(X),
            -1,
            dtype=np.int32
        )

    return (
        X,
        y,
        timestamps
    )


# ============================================================
# SCAN ONE SPLIT
# ============================================================

def scan_split(
    split_root
):

    """
    Scan:
        train
        dev
        eval

    Returns a list of recording dictionaries.
    """

    if not os.path.exists(
        split_root
    ):

        raise FileNotFoundError(
            f"Split directory not found:\n"
            f"{split_root}"
        )

    edfs = glob.glob(
        os.path.join(
            split_root,
            "**",
            "*.edf"
        ),
        recursive=True
    )

    edfs.sort()

    records = []

    for edf_path in edfs:

        annotation = find_annotation_file(
            edf_path
        )

        seizures = parse_annotation_file(
            annotation
        )

        rel = os.path.relpath(
            edf_path,
            split_root
        )

        parts = rel.split(
            os.sep
        )

        patient = (
            parts[0]
            if len(parts) >= 1
            else "unknown"
        )

        session = (
            parts[1]
            if len(parts) >= 2
            else "unknown"
        )

        montage = (
            parts[2]
            if len(parts) >= 3
            else "unknown"
        )

        records.append({

            "path":
                edf_path,

            "annotation":
                annotation,

            "seizures":
                seizures,

            "patient":
                patient,

            "session":
                session,

            "montage":
                montage,

            "split":
                os.path.basename(
                    split_root
                )

        })

    return records


# ============================================================
# DATASET SCAN
# ============================================================

def scan_tusz_dataset():

    print(
        "\n=================================================="
    )

    print(
        "TUSZ v2.0.3 DATASET SCAN"
    )

    print(
        "=================================================="
    )

    print(
        f"\nDataset root:"
    )

    print(
        DATASET_ROOT
    )

    train_records = scan_split(
        TRAIN_ROOT
    )

    dev_records = scan_split(
        DEV_ROOT
    )

    eval_records = scan_split(
        EVAL_ROOT
    )

    print(
        f"\nTrain EDF files: "
        f"{len(train_records)}"
    )

    print(
        f"Dev EDF files:   "
        f"{len(dev_records)}"
    )

    print(
        f"Eval EDF files:  "
        f"{len(eval_records)}"
    )

    print(
        f"Total EDF files: "
        f"{len(train_records) + len(dev_records) + len(eval_records)}"
    )

    return (
        train_records,
        dev_records,
        eval_records
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_split_summary(
    records,
    name
):

    seizure_files = 0

    seizure_events = 0

    patients = set()

    sessions = set()

    for record in records:

        patients.add(
            record["patient"]
        )

        sessions.add(
            (
                record["patient"],
                record["session"]
            )
        )

        if record["seizures"]:

            seizure_files += 1

            seizure_events += len(
                record["seizures"]
            )

    print(
        f"\n{name.upper()} SUMMARY"
    )

    print(
        "--------------------------------------------------"
    )

    print(
        f"EDF files       : {len(records)}"
    )

    print(
        f"Patients        : {len(patients)}"
    )

    print(
        f"Sessions        : {len(sessions)}"
    )

    print(
        f"Seizure files   : {seizure_files}"
    )

    print(
        f"Seizure events  : {seizure_events}"
    )


# ============================================================
# PREPARE RECORDING DATASET
# ============================================================

def prepare_records(
    records,
    split_name
):

    """
    Convert all EDF recordings into window arrays.

    IMPORTANT:
    This is memory-heavy for the complete TUSZ corpus.
    """

    all_X = []

    all_y = []

    print(
        "\n=================================================="
    )

    print(
        f"PREPARING {split_name.upper()} DATASET"
    )

    print(
        "=================================================="
    )

    total = len(
        records
    )

    for index, record in enumerate(
        records,
        start=1
    ):

        edf_path = record["path"]

        print(
            f"\n[{index}/{total}] "
            f"{os.path.basename(edf_path)}",
            flush=True
        )

        try:

            X_part, y_part, _ = (
                process_edf_into_windows(
                    edf_path,
                    record["seizures"],
                    create_labels=True
                )
            )

            if len(X_part) == 0:

                print(
                    "  No windows extracted."
                )

                continue

            all_X.append(
                X_part
            )

            all_y.append(
                y_part
            )

            print(
                f"  Extracted windows: "
                f"{len(X_part)}"
            )

        except Exception as e:

            print(
                f"  ERROR processing EDF:"
                f" {e}"
            )

    if not all_X:

        raise RuntimeError(
            f"No windows were extracted "
            f"for {split_name}."
        )

    X = np.concatenate(
        all_X,
        axis=0
    )

    y = np.concatenate(
        all_y,
        axis=0
    )

    print(
        "\n=================================================="
    )

    print(
        f"{split_name.upper()} DATASET READY"
    )

    print(
        "=================================================="
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    for c in range(
        NUM_CLASSES
    ):

        count = int(
            np.sum(
                y == c
            )
        )

        percentage = (
            count /
            len(y) *
            100
        )

        print(
            f"{LABEL_NAMES[c]:15s}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    return (
        X,
        y
    )


# ============================================================
# MODEL
# ============================================================

def build_tensorflow_model():
    """
    Deeper 1D CNN for 3-class EEG classification.
    Input: 10 seconds x 128 Hz x 22 TCP channels.
    Output: Normal / Pre-Seizure / Seizure.
    """
    model = models.Sequential([
        layers.Input(shape=(SAMPLES_PER_WINDOW, NUM_CHANNELS)),

        # Block 1
        layers.Conv1D(64, 7, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv1D(64, 7, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.15),

        # Block 2
        layers.Conv1D(128, 5, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv1D(128, 5, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.20),

        # Block 3
        layers.Conv1D(256, 5, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv1D(256, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv1D(384, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv1D(384, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.30),

        # Block 5
        layers.Conv1D(512, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv1D(512, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.35),

        # Classifier
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.40),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ], name="TUSZ_EEG_DEEP_CNN")

    optimizer = keras.optimizers.Adam(
        learning_rate=3e-4,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ============================================================
# MODEL ARCHITECTURE IMAGE
# ============================================================

def save_model_architecture(
    model
):

    if not MATPLOTLIB_AVAILABLE:

        return

    try:

        keras.utils.plot_model(

            model,

            to_file=
            MODEL_IMAGE_PATH,

            show_shapes=True,

            show_layer_names=True,

            expand_nested=True,

            dpi=150
        )

        print(
            f"\nModel architecture saved:"
            f"\n{MODEL_IMAGE_PATH}"
        )

    except Exception as e:

        print(
            "\nModel architecture image "
            "could not be generated."
        )

        print(
            f"Reason: {e}"
        )

        print(
            "You may need pydot and Graphviz."
        )


# ============================================================
# TRAINING CURVES
# ============================================================

def plot_training_curves(
    history
):

    if not MATPLOTLIB_AVAILABLE:

        return

    hist = history.history

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    if "loss" in hist:

        ax.plot(
            hist["loss"],
            label="Training Loss",
            lw=2
        )

    if "val_loss" in hist:

        ax.plot(
            hist["val_loss"],
            label="Validation Loss",
            lw=2,
            linestyle="--"
        )

    ax.set_title(
        "TUSZ EEG Training Loss"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        TRAINING_LOSS_PATH,
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    if "accuracy" in hist:

        ax.plot(
            np.asarray(
                hist["accuracy"]
            ) * 100,
            label="Training Accuracy",
            lw=2
        )

    if "val_accuracy" in hist:

        ax.plot(
            np.asarray(
                hist["val_accuracy"]
            ) * 100,
            label="Validation Accuracy",
            lw=2,
            linestyle="--"
        )

    ax.set_title(
        "TUSZ EEG Training Accuracy"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        TRAINING_ACC_PATH,
        dpi=300
    )

    plt.close()

    print(
        "\nTraining graphs saved."
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    cm,
    output_path,
    title
):

    if not MATPLOTLIB_AVAILABLE:

        return

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_xticks(
        range(NUM_CLASSES)
    )

    ax.set_yticks(
        range(NUM_CLASSES)
    )

    ax.set_xticklabels(
        [
            LABEL_NAMES[i]
            for i in range(NUM_CLASSES)
        ],
        rotation=20
    )

    ax.set_yticklabels(
        [
            LABEL_NAMES[i]
            for i in range(NUM_CLASSES)
        ]
    )

    threshold = (
        cm.max() / 2
        if cm.size
        else 0
    )

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            NUM_CLASSES
        ):

            value = cm[i, j]

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
                fontsize=12,
                fontweight="bold"
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved:"
        f"\n{output_path}"
    )


# ============================================================
# ROC CURVE
# ============================================================

def save_roc_curve(
    y_true,
    y_prob,
    output_path,
    title
):

    if not MATPLOTLIB_AVAILABLE:

        return {}

    if not SKLEARN_AVAILABLE:

        return {}

    y_binary = label_binarize(
        y_true,
        classes=np.arange(
            NUM_CLASSES
        )
    )

    roc_values = {}

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    for class_id in range(
        NUM_CLASSES
    ):

        positive = np.sum(
            y_binary[:, class_id] == 1
        )

        negative = np.sum(
            y_binary[:, class_id] == 0
        )

        if (
            positive == 0
            or
            negative == 0
        ):

            continue

        fpr, tpr, _ = roc_curve(

            y_binary[:, class_id],

            y_prob[:, class_id]
        )

        class_auc = auc(
            fpr,
            tpr
        )

        roc_values[
            LABEL_NAMES[class_id]
        ] = float(
            class_auc
        )

        ax.plot(

            fpr,

            tpr,

            lw=2,

            label=(
                f"{LABEL_NAMES[class_id]} "
                f"(AUC={class_auc:.4f})"
            )
        )

    ax.plot(

        [0, 1],

        [0, 1],

        linestyle="--",

        lw=1.5,

        label="Random Classifier"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Macro AUC
    # --------------------------------------------------------

    try:

        macro_auc = roc_auc_score(

            y_binary,

            y_prob,

            average="macro"
        )

        roc_values[
            "Macro ROC-AUC"
        ] = float(
            macro_auc
        )

    except Exception:

        pass

    print(
        f"ROC curve saved:"
        f"\n{output_path}"
    )

    return roc_values


# ============================================================
# SAVE EVALUATION FILES
# ============================================================

def save_evaluation_files(

    y_true,

    y_pred,

    y_prob,

    evaluation_name,

    confusion_path,

    roc_path,

    report_path,

    json_path
):

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=np.arange(
            NUM_CLASSES
        )
    )

    accuracy = float(
        np.mean(
            y_true == y_pred
        )
    )

    report = classification_report(

        y_true,

        y_pred,

        labels=np.arange(
            NUM_CLASSES
        ),

        target_names=[
            LABEL_NAMES[i]
            for i in range(NUM_CLASSES)
        ],

        zero_division=0
    )

    report_dict = classification_report(

        y_true,

        y_pred,

        labels=np.arange(
            NUM_CLASSES
        ),

        target_names=[
            LABEL_NAMES[i]
            for i in range(NUM_CLASSES)
        ],

        zero_division=0,

        output_dict=True
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(

            y_true,

            y_pred,

            labels=np.arange(
                NUM_CLASSES
            ),

            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    save_confusion_matrix(

        cm,

        confusion_path,

        f"TUSZ EEG {evaluation_name} "
        "Confusion Matrix"
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    roc_values = save_roc_curve(

        y_true,

        y_prob,

        roc_path,

        f"TUSZ EEG {evaluation_name} "
        "ROC Curve"
    )

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------

    lines = []

    lines.append(
        "============================================================"
    )

    lines.append(
        "TUSZ v2.0.3 EEG CLASSIFICATION REPORT"
    )

    lines.append(
        "============================================================"
    )

    lines.append("")

    lines.append(
        f"Evaluation Set: {evaluation_name}"
    )

    lines.append(
        f"Model: TensorFlow 1D CNN"
    )

    lines.append(
        f"Input Shape: "
        f"({SAMPLES_PER_WINDOW}, {NUM_CHANNELS})"
    )

    lines.append(
        f"Sampling Frequency: "
        f"{TARGET_FS} Hz"
    )

    lines.append(
        f"Window Duration: "
        f"{WINDOW_SIZE_SEC} seconds"
    )

    lines.append(
        f"Pre-Seizure Duration: "
        f"{PREDICTION_TIME_SEC / 60:.0f} minutes"
    )

    lines.append(
        f"Total Evaluation Windows: "
        f"{len(y_true)}"
    )

    lines.append("")

    lines.append(
        "OVERALL ACCURACY"
    )

    lines.append(
        "------------------------------------------------------------"
    )

    lines.append(
        f"{accuracy * 100:.4f}%"
    )

    lines.append("")

    lines.append(
        "CLASSIFICATION REPORT"
    )

    lines.append(
        "------------------------------------------------------------"
    )

    lines.append(
        report
    )

    lines.append(
        "CLASS-WISE METRICS"
    )

    lines.append(
        "------------------------------------------------------------"
    )

    for i in range(
        NUM_CLASSES
    ):

        lines.append(
            f"{LABEL_NAMES[i]}:"
        )

        lines.append(
            f"  Precision = "
            f"{precision[i]:.6f}"
        )

        lines.append(
            f"  Recall    = "
            f"{recall[i]:.6f}"
        )

        lines.append(
            f"  F1 Score  = "
            f"{f1[i]:.6f}"
        )

        lines.append(
            f"  Support   = "
            f"{int(support[i])}"
        )

        lines.append("")

    lines.append(
        "CONFUSION MATRIX"
    )

    lines.append(
        "------------------------------------------------------------"
    )

    lines.append(
        str(cm)
    )

    lines.append("")

    lines.append(
        "ROC-AUC RESULTS"
    )

    lines.append(
        "------------------------------------------------------------"
    )

    for name, value in roc_values.items():

        lines.append(
            f"{name}: {value:.6f}"
        )

    lines.append("")

    lines.append(
        "============================================================"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(
                lines
            )
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_data = {

        "dataset":
            "TUSZ v2.0.3",

        "evaluation_set":
            evaluation_name,

        "model":
            "TensorFlow 1D CNN",

        "input_shape": [

            SAMPLES_PER_WINDOW,

            NUM_CHANNELS
        ],

        "sampling_frequency_hz":
            TARGET_FS,

        "window_duration_seconds":
            WINDOW_SIZE_SEC,

        "pre_seizure_duration_seconds":
            PREDICTION_TIME_SEC,

        "classes":
            LABEL_NAMES,

        "total_windows":
            int(len(y_true)),

        "accuracy":
            accuracy,

        "accuracy_percent":
            accuracy * 100,

        "classification_report":
            report_dict,

        "confusion_matrix":
            cm.tolist(),

        "roc_auc":
            roc_values,

        "output_files": {

            "confusion_matrix":
                confusion_path,

            "roc_curve":
                roc_path,

            "classification_report":
                report_path

        }
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_data,
            f,
            indent=4
        )

    print(
        f"\nClassification report saved:"
        f"\n{report_path}"
    )

    print(
        f"Metrics JSON saved:"
        f"\n{json_path}"
    )

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "roc_auc": roc_values
    }


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    evaluation_name
):

    print(
        "\n=================================================="
    )

    print(
        f"EVALUATING {evaluation_name.upper()} SET"
    )

    print(
        "=================================================="
    )

    probabilities = model.predict(
        X,
        batch_size=32,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    if evaluation_name.lower() == "dev":

        confusion_path = (
            DEV_CONFUSION_MATRIX_PATH
        )

        roc_path = (
            DEV_ROC_CURVE_PATH
        )

        report_path = (
            DEV_CLASSIFICATION_REPORT_PATH
        )

        json_path = (
            DEV_METRICS_JSON_PATH
        )

    else:

        confusion_path = (
            EVAL_CONFUSION_MATRIX_PATH
        )

        roc_path = (
            EVAL_ROC_CURVE_PATH
        )

        report_path = (
            EVAL_CLASSIFICATION_REPORT_PATH
        )

        json_path = (
            EVAL_METRICS_JSON_PATH
        )

    results = save_evaluation_files(

        y,

        predictions,

        probabilities,

        evaluation_name,

        confusion_path,

        roc_path,

        report_path,

        json_path
    )

    print(
        "\n=================================================="
    )

    print(
        f"{evaluation_name.upper()} RESULTS"
    )

    print(
        "=================================================="
    )

    print(
        f"Accuracy: "
        f"{results['accuracy'] * 100:.4f}%"
    )

    print(
        "\nROC-AUC:"
    )

    for name, value in (
        results["roc_auc"].items()
    ):

        print(
            f"  {name}: {value:.6f}"
        )

    return results


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
    X_dev,
    y_dev,
    epochs=100,
    batch_size=32
):

    print(
        "\n=================================================="
    )

    print(
        "BUILDING TUSZ TENSORFLOW MODEL"
    )

    print(
        "=================================================="
    )

    model = build_tensorflow_model()

    model.summary()

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(

        class_weight="balanced",

        classes=classes,

        y=y_train
    )

    class_weights = {

        int(c):
            float(w)

        for c, w in zip(
            classes,
            weights
        )
    }

    print(
        "\nClass weights:"
    )

    print(
        class_weights
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callback_list = [

        callbacks.ModelCheckpoint(

            MODEL_SAVE_PATH,

            monitor="val_loss",

            save_best_only=True,

            verbose=1
        ),

        callbacks.EarlyStopping(

            monitor="val_loss",

            patience=7,

            restore_best_weights=True,

            verbose=1
        ),

        callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=3,

            verbose=1
        )

    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\n=================================================="
    )

    print(
        "STARTING TUSZ TRAINING"
    )

    print(
        "=================================================="
    )

    history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_dev,

            y_dev
        ),

        epochs=epochs,

        batch_size=batch_size,

        class_weight=class_weights,

        callbacks=callback_list,

        verbose=1
    )

    print(
        "\nTraining completed."
    )

    best_val_accuracy = max(
        [float(v) for v in history.history.get("val_accuracy", [0.0])]
    )
    best_train_accuracy = max(
        [float(v) for v in history.history.get("accuracy", [0.0])]
    )

    print(
        f"\nBest training accuracy: {best_train_accuracy * 100:.2f}%"
    )
    print(
        f"Best validation accuracy: {best_val_accuracy * 100:.2f}%"
    )

    if best_val_accuracy >= 0.90:
        print("90% validation-accuracy target reached.")
    else:
        print(
            "90% validation-accuracy target was not reached in this run. "
            "The code cannot honestly guarantee 90%; the achieved value depends "
            "on the data, labels, preprocessing and training run."
        )

    print(
        f"Model saved:"
        f"\n{MODEL_SAVE_PATH}"
    )

    save_model_architecture(
        model
    )

    plot_training_curves(
        history
    )

    return model


# ============================================================
# USER EDF FILE SELECTION
# ============================================================

def select_user_edf():

    """
    Open a graphical file picker.

    User selects any EDF file.
    """

    if not TKINTER_AVAILABLE:

        print(
            "Tkinter is unavailable."
        )

        path = input(
            "\nEnter EDF file path manually: "
        ).strip()

        return path

    root = tk.Tk()

    root.withdraw()

    root.attributes(
        "-topmost",
        True
    )

    file_path = filedialog.askopenfilename(

        title=(
            "Select EEG EDF File "
            "for Prediction"
        ),

        filetypes=[

            (
                "EDF files",
                "*.edf"
            ),

            (
                "All files",
                "*.*"
            )

        ]
    )

    root.destroy()

    return file_path


# ============================================================
# USER EDF PREDICTION
# ============================================================

def predict_user_edf(
    model,
    edf_path=None
):

    """
    Predict a user-provided EDF.

    IMPORTANT:

    No .csv or .csv_bi annotation is required.

    The model analyzes the raw EEG signal.
    """

    if edf_path is None:

        edf_path = select_user_edf()

    if not edf_path:

        print(
            "\nNo EDF file selected."
        )

        return None

    if not os.path.exists(
        edf_path
    ):

        print(
            "\nEDF file does not exist:"
        )

        print(
            edf_path
        )

        return None

    if not edf_path.lower().endswith(
        ".edf"
    ):

        print(
            "\nSelected file is not an EDF."
        )

        return None

    print(
        "\n=================================================="
    )

    print(
        "USER EDF PREDICTION"
    )

    print(
        "=================================================="
    )

    print(
        f"\nSelected EDF:"
    )

    print(
        edf_path
    )

    # --------------------------------------------------------
    # Process WITHOUT annotations
    # --------------------------------------------------------

    X, _, timestamps = (
        process_edf_into_windows(

            edf_path,

            seizures=[],

            create_labels=False
        )
    )

    if len(X) == 0:

        print(
            "\nNo valid 10-second EEG windows "
            "could be extracted."
        )

        return None

    print(
        f"\nTotal EEG windows: "
        f"{len(X)}"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    probabilities = model.predict(

        X,

        batch_size=32,

        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Window class counts
    # --------------------------------------------------------

    normal_count = int(
        np.sum(
            predictions == 0
        )
    )

    pre_count = int(
        np.sum(
            predictions == 1
        )
    )

    seizure_count = int(
        np.sum(
            predictions == 2
        )
    )

    # --------------------------------------------------------
    # Average probabilities
    # --------------------------------------------------------

    average_probability = np.mean(
        probabilities,
        axis=0
    )

    # --------------------------------------------------------
    # Overall prediction
    # --------------------------------------------------------

    overall_class = int(
        np.argmax(
            average_probability
        )
    )

    overall_label = (
        LABEL_NAMES[
            overall_class
        ]
    )

    # --------------------------------------------------------
    # Sustained seizure detection
    # --------------------------------------------------------

    sustained_events = []

    consecutive = 0

    event_start = None

    for i, prediction in enumerate(
        predictions
    ):

        if prediction == 2:

            if consecutive == 0:

                event_start = (
                    timestamps[i][0]
                )

            consecutive += 1

        else:

            if (
                consecutive >= CONSECUTIVE_SEIZURE_WINDOWS_REQUIRED
            ):
                sustained_events.append(

                    (

                        event_start,

                        timestamps[i - 1][1],

                        consecutive
                    )
                )

            consecutive = 0

            event_start = None

    if (
        consecutive
        >=
        CONSECUTIVE_SEIZURE_WINDOWS_REQUIRED
    ):

        sustained_events.append(

            (

                event_start,

                timestamps[-1][1],

                consecutive
            )
        )

    # --------------------------------------------------------
    # Print prediction
    # --------------------------------------------------------

    print(
        "\n=================================================="
    )

    print(
        "FINAL EDF PREDICTION"
    )

    print(
        "=================================================="
    )

    print(
        f"\nPrediction:"
    )

    print(
        f"  {overall_label}"
    )

    print(
        "\nAverage model probabilities:"
    )

    print(
        f"  Normal      : "
        f"{average_probability[0] * 100:.2f}%"
    )

    print(
        f"  Pre-Seizure : "
        f"{average_probability[1] * 100:.2f}%"
    )

    print(
        f"  Seizure     : "
        f"{average_probability[2] * 100:.2f}%"
    )

    print(
        "\nWindow predictions:"
    )

    print(
        f"  Normal      : "
        f"{normal_count}"
    )

    print(
        f"  Pre-Seizure : "
        f"{pre_count}"
    )

    print(
        f"  Seizure     : "
        f"{seizure_count}"
    )

    print(
        "\nSustained seizure events:"
    )

    if sustained_events:

        for i, (
            start,
            end,
            count
        ) in enumerate(
            sustained_events,
            start=1
        ):

            print(
                f"  Event {i}: "
                f"{format_time(start)} -> "
                f"{format_time(end)} "
                f"({count} consecutive windows)"
            )

    else:

        print(
            "  No sustained seizure event detected."
        )

    # --------------------------------------------------------
    # Save TXT report
    # --------------------------------------------------------

    lines = []

    lines.append(
        "============================================================"
    )

    lines.append(
        "TUSZ EEG USER EDF PREDICTION REPORT"
    )

    lines.append(
        "============================================================"
    )

    lines.append("")

    lines.append(
        f"EDF File:"
    )

    lines.append(
        os.path.basename(
            edf_path
        )
    )

    lines.append("")

    lines.append(
        f"Full Path:"
    )

    lines.append(
        edf_path
    )

    lines.append("")

    lines.append(
        "MODEL INFORMATION"
    )

    lines.append(
        "------------------------------------------------------------"
    )

    lines.append(
        "Model: TensorFlow 1D CNN"
    )

    lines.append(
        f"Input Shape: "
        f"({SAMPLES_PER_WINDOW}, {NUM_CHANNELS})"
    )

    lines.append(
        f"Sampling Frequency: "
        f"{TARGET_FS} Hz"
    )

    lines.append(
        f"Window Size: "
        f"{WINDOW_SIZE_SEC} seconds"
    )

    lines.append(
        "TCP Channels: 22"
    )

    lines.append("")

    lines.append(
        "PREDICTION"
    )

    lines.append(
        "------------------------------------------------------------"
    )

    lines.append(
        f"Final Prediction: "
        f"{overall_label}"
    )

    lines.append("")

    lines.append(
        "AVERAGE PROBABILITIES"
    )

    lines.append(
        "------------------------------------------------------------"
    )

    lines.append(
        f"Normal: "
        f"{average_probability[0] * 100:.4f}%"
    )

    lines.append(
        f"Pre-Seizure: "
        f"{average_probability[1] * 100:.4f}%"
    )

    lines.append(
        f"Seizure: "
        f"{average_probability[2] * 100:.4f}%"
    )

    lines.append("")

    lines.append(
        "WINDOW COUNTS"
    )

    lines.append(
        "------------------------------------------------------------"
    )

    lines.append(
        f"Normal: {normal_count}"
    )

    lines.append(
        f"Pre-Seizure: {pre_count}"
    )

    lines.append(
        f"Seizure: {seizure_count}"
    )

    lines.append(
        f"Total: {len(X)}"
    )

    lines.append("")

    lines.append(
        "SUSTAINED SEIZURE EVENTS"
    )

    lines.append(
        "------------------------------------------------------------"
    )

    if sustained_events:

        for i, (
            start,
            end,
            count
        ) in enumerate(
            sustained_events,
            start=1
        ):

            lines.append(
                f"Event {i}: "
                f"{format_time(start)} -> "
                f"{format_time(end)} | "
                f"{count} windows"
            )

    else:

        lines.append(
            "No sustained seizure events detected."
        )

    lines.append("")

    lines.append(
        "IMPORTANT NOTICE"
    )

    lines.append(
        "------------------------------------------------------------"
    )

    lines.append(
        "This is an experimental machine-learning "
        "prediction system."
    )

    lines.append(
        "It is not a medical diagnosis and should "
        "not replace qualified clinical interpretation."
    )

    lines.append("")

    lines.append(
        "============================================================"
    )

    with open(
        USER_EDF_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(
                lines
            )
        )

    print(
        f"\nPrediction report saved:"
        f"\n{USER_EDF_REPORT_PATH}"
    )

    return {

        "edf":
            edf_path,

        "prediction":
            overall_label,

        "prediction_class":
            overall_class,

        "probabilities": {

            "Normal":
                float(
                    average_probability[0]
                ),

            "Pre-Seizure":
                float(
                    average_probability[1]
                ),

            "Seizure":
                float(
                    average_probability[2]
                )
        },

        "window_counts": {

            "Normal":
                normal_count,

            "Pre-Seizure":
                pre_count,

            "Seizure":
                seizure_count
        },

        "sustained_seizure_events":
            sustained_events
    }


# ============================================================
# LOAD EXISTING TRAINED MODEL
# ============================================================

def load_existing_model():
    """
    Load the previously trained TUSZ 3-class EEG CNN model.

    The model is loaded with compile=False because prediction does not
    require optimizer/loss state. This also makes loading more robust
    across Keras 3 / backend configurations.
    """

    if not TF_AVAILABLE:
        print("\nERROR: Keras/deep-learning backend is not available.")
        print("Install the required deep-learning dependencies first.")
        return None

    if not os.path.isfile(MODEL_SAVE_PATH):
        print("\nERROR: Trained model not found.")
        print(f"Expected model:\n{MODEL_SAVE_PATH}")
        print("\nTrain the model first using option 1.")
        return None

    print("\n==================================================")
    print("LOADING TRAINED MODEL")
    print("==================================================")
    print(f"Model:\n{MODEL_SAVE_PATH}")

    try:
        model = keras.models.load_model(
            MODEL_SAVE_PATH,
            compile=False
        )

        print("\nModel loaded successfully.")

        try:
            print(f"Input shape  : {model.input_shape}")
            print(f"Output shape : {model.output_shape}")
        except Exception:
            pass

        return model

    except Exception as e:
        print("\nERROR: Could not load the trained model.")
        print(f"{type(e).__name__}: {e}")
        print("\nMake sure the .h5 model was trained with the same model/preprocessing configuration as this script.")
        return None


# ============================================================
# MENU
# ============================================================

def show_menu():
    print(
        "\n=================================================="
    )
    print(
        "TUSZ EEG DEEP LEARNING SYSTEM"
    )
    print(
        "=================================================="
    )
    print(
        "\n1. Train Model"
    )
    print(
        "2. Use Trained Model - Predict User EDF"
    )
    print(
        "3. Exit"
    )
    print(
        "=================================================="
    )


def complete_training_and_evaluation_workflow(
    train_records,
    dev_records,
    eval_records
):
    """
    Option 1:
      1) Prepare train data
      2) Prepare dev data
      3) Prepare eval data
      4) Train deep CNN
      5) Evaluate dev
      6) Evaluate eval
      7) Save all reports/plots/model
    """
    X_train, y_train = prepare_records(
        train_records,
        "train"
    )

    X_dev, y_dev = prepare_records(
        dev_records,
        "dev"
    )

    model = train_model(
        X_train,
        y_train,
        X_dev,
        y_dev,
        epochs=50,
        batch_size=32
    )

    print(
        "\n=================================================="
    )
    print(
        "DEV DATASET EVALUATION"
    )
    print(
        "=================================================="
    )

    dev_results = evaluate_model(
        model,
        X_dev,
        y_dev,
        "dev"
    )

    # Release training/dev arrays before loading eval where possible.
    del X_train
    del y_train

    print(
        "\n=================================================="
    )
    print(
        "PREPARING EVAL DATASET"
    )
    print(
        "=================================================="
    )

    X_eval, y_eval = prepare_records(
        eval_records,
        "eval"
    )

    print(
        "\n=================================================="
    )
    print(
        "BLIND EVALUATION DATASET"
    )
    print(
        "=================================================="
    )

    eval_results = evaluate_model(
        model,
        X_eval,
        y_eval,
        "eval"
    )

    print(
        "\n=================================================="
    )
    print(
        "TRAINING + EVALUATION COMPLETE"
    )
    print(
        "=================================================="
    )
    print(
        f"Dev accuracy : {dev_results['accuracy'] * 100:.2f}%"
    )
    print(
        f"Eval accuracy: {eval_results['accuracy'] * 100:.2f}%"
    )
    print(
        "\nAll model, graph, confusion-matrix, ROC and report files "
        f"are saved in:\n{RESULTS_DIR}"
    )

    return model


def main():
    print(
        "\n============================================================"
    )
    print(
        "TUSZ v2.0.3"
    )
    print(
        "TEMPLE UNIVERSITY HOSPITAL EEG"
    )
    print(
        "SEIZURE DETECTION + PRE-SEIZURE PREDICTION"
    )
    print(
        "DEEP 1D CNN - 3 CLASS SYSTEM"
    )
    print(
        "============================================================"
    )

    print(
        f"\nAcceleration Device:\n  {ACTIVE_BACKEND}"
    )
    print(
        f"\nDataset root:\n{DATASET_ROOT}"
    )
    print(
        f"\nModel:\n{MODEL_SAVE_PATH}"
    )
    print(
        f"\nResults:\n{RESULTS_DIR}"
    )

    # Dependency checks
    if not TF_AVAILABLE:
        print("\nTensorFlow/Keras is required.")
        return

    if not SKLEARN_AVAILABLE:
        print("\nScikit-learn is required.")
        return

    if not MNE_AVAILABLE:
        print("\nMNE is required.")
        return

    if not os.path.exists(DATASET_ROOT):
        print("\nDataset root does not exist:")
        print(DATASET_ROOT)
        return

    if not os.path.exists(TRAIN_ROOT):
        print("\nTrain directory not found:")
        print(TRAIN_ROOT)
        print(
            "\nExpected structure: <dataset_root>/edf/train, "
            "<dataset_root>/edf/dev, <dataset_root>/edf/eval"
        )
        return

    # Scan once when the program starts.
    (
        train_records,
        dev_records,
        eval_records
    ) = scan_tusz_dataset()

    print_split_summary(
        train_records,
        "train"
    )
    print_split_summary(
        dev_records,
        "dev"
    )
    print_split_summary(
        eval_records,
        "eval"
    )

    model = None

    while True:
        show_menu()

        choice = input(
            "\nEnter your choice: "
        ).strip()

        # ========================================================
        # 1. TRAIN EVERYTHING
        # ========================================================
        if choice == "1":
            try:
                model = complete_training_and_evaluation_workflow(
                    train_records,
                    dev_records,
                    eval_records
                )
            except KeyboardInterrupt:
                print("\nTraining interrupted by user.")
            except Exception as e:
                print(
                    "\nTraining/evaluation failed:"
                )
                print(
                    f"{type(e).__name__}: {e}"
                )

        # ========================================================
        # 2. USER EDF PREDICTION
        # ========================================================
        elif choice == "2":
            if model is None:
                model = load_existing_model()

            if model is None:
                print(
                    "\nNo trained model was found."
                )
                print(
                    "Choose option 1 first to train the model."
                )
                continue

            try:
                predict_user_edf(
                    model
                )
            except KeyboardInterrupt:
                print("\nPrediction interrupted by user.")
            except Exception as e:
                print(
                    "\nEDF prediction failed:"
                )
                print(
                    f"{type(e).__name__}: {e}"
                )

        # ========================================================
        # 3. EXIT
        # ========================================================
        elif choice == "3":
            print(
                "\nExiting TUSZ EEG system."
            )
            break

        else:
            print(
                "\nInvalid choice. Please enter 1, 2 or 3."
            )


if __name__ == "__main__":
    main()
