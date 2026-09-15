"""
Stub model generator for NeuroTrace backend development/testing.

Generates tusz_3class_eeg_model.h5 with the EXACT architecture and
input shape from python code.py so the API starts correctly.

Usage:
    pip install tensorflow numpy
    python create_stub_model.py

The saved model is binary-compatible with keras.models.load_model().
Drop your real model at model/tusz_3class_eeg_model.h5 when ready.
"""

import os
import numpy as np
from pathlib import Path

# ====================================================================
# Constants extracted verbatim from python code.py
# ====================================================================
TARGET_FS        = 128    # Hz
WINDOW_SIZE_SEC  = 10     # seconds
SAMPLES_PER_WIN  = TARGET_FS * WINDOW_SIZE_SEC  # 1280
NUM_CHANNELS     = 22     # 22-channel TUSZ TCP montage
NUM_CLASSES      = 3      # Normal=0, Pre-Seizure=1, Seizure=2

MODEL_DIR  = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "tusz_3class_eeg_model.h5"


def build_stub_model():
    """Exact same architecture as build_tensorflow_model() in python code.py."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(SAMPLES_PER_WIN, NUM_CHANNELS)),

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
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="TUSZ_EEG_DEEP_CNN")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    if MODEL_PATH.exists():
        print(f"Model already exists: {MODEL_PATH}")
        print("Delete it first if you want to regenerate.")
        return

    print("Building stub TUSZ 1D-CNN …")
    model = build_stub_model()
    model.summary()

    print(f"\nInput  shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    print(f"\nSaving → {MODEL_PATH}")
    model.save(str(MODEL_PATH))
    print("Done. Start the server and upload an EDF to test the full pipeline.")
    print("\nNOTE: Replace this stub with your real trained model for actual predictions.")


if __name__ == "__main__":
    main()
