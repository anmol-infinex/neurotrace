# PRD.md — NeuroTrace: EEG Seizure Prediction Platform

## 1. Overview

NeuroTrace is a full-stack web application that lets a user upload a raw
`.edf` EEG recording and receive an automated seizure-risk assessment: a
final classification (Normal / Pre-Seizure / Seizure), per-class
probabilities, a windowed timeline of sustained seizure events, an 18-channel
signal visualization, and (when ground truth is available) model evaluation
artifacts (ROC curve, confusion matrix). The existing frontend
(`neurotrace-dashboard-eight.vercel.app`) currently does local, client-side
EDF parsing only; this PRD covers turning it into a real product backed by
a trained Keras model served from a dedicated backend.

## 2. Problem Statement

Reviewing raw EEG for seizure activity is slow and requires a trained
neurologist. A lightweight, automated triage layer — "does this recording
look normal, pre-ictal, or ictal, and where" — can speed up review and
serve as a portfolio-grade demonstration of an applied AI/ML + full-stack
+ security-aware engineering pipeline (TUSZ dataset, deployed inference
service, clinical-style reporting).

## 3. Goals

- Let a user upload a `.edf` file and get a model prediction back through a
  real backend (not client-side JS).
- Match the exact report format already specified (see `design.md` §3):
  final prediction, average probabilities, window counts, sustained seizure
  events.
- Visualize the recording as an 18-channel EEG trace.
- Support downloadable outputs: text report, JSON, ROC curve PNG, confusion
  matrix PNG + text.
- Deploy end-to-end: Vercel (frontend) talking to a hosted FastAPI backend.

## 4. Non-Goals (out of scope for v1)

- Clinical/diagnostic use — this is explicitly a research/demo prototype
  (already stated on the live UI).
- Real-time streaming EEG ingestion from hardware (this PRD covers file
  upload only; the earlier Seizure-Sentry wearable concept is a separate,
  larger project — see `[[seizure-sentry]]`).
- Multi-user auth / patient record management (v1 is single-session,
  upload-and-review).
- Model retraining pipeline (model is already trained; backend only serves
  it).

## 5. Users

- **Primary:** the developer (Anmol) — for grading, portfolio, and
  demonstrating an applied AI security/assurance-adjacent skillset.
- **Secondary:** anyone reviewing the project (professor, recruiter,
  interviewer) uploading a sample `.edf` to see the pipeline work.

## 6. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-1 | User can upload a `.edf` file via the existing NeuroTrace UI. |
| FR-2 | Backend parses the EDF, selects/validates the 18-channel montage, resamples to the model's expected rate. |
| FR-3 | Backend windows the signal and runs the Keras model on each window. |
| FR-4 | Backend returns: final prediction, average class probabilities, per-class window counts, total windows, sustained seizure event list (start/end/window count). |
| FR-5 | Backend returns a downsampled multi-channel signal payload for the frontend to render as an 18-channel graph. |
| FR-6 | If a ground-truth annotation file is supplied, backend computes and returns ROC curve (PNG) and confusion matrix (PNG + text). |
| FR-7 | User can download: text report, JSON result, ROC PNG, confusion matrix PNG, confusion matrix text. |
| FR-8 | Frontend displays a clear "prototype, not for clinical use" disclaimer (already present). |

## 7. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | Backend must reject non-`.edf` uploads with a clear error. |
| NFR-2 | Inference on a typical single-session TUSZ file should complete in well under 30s on the hosting tier used. |
| NFR-3 | CORS locked to the known frontend origin(s), not wildcard, since this is a public demo endpoint. |
| NFR-4 | Uploaded EDFs and generated artifacts are per-job (UUID-scoped) and not mixed between users. |
| NFR-5 | No PHI/real patient data — TUSZ is a research corpus; document this clearly if the demo is shared publicly. |

## 8. Success Criteria

- End-to-end demo works live: upload `.edf` on the Vercel site → real model
  prediction from the hosted backend → report + graph render → all
  downloads work.
- Prediction report text output matches the specified format exactly.
- Deployed and reachable via a public URL for grading/demo purposes.

## 9. Risks / Open Questions

- Exact preprocessing (sample rate, window size, channel order, input
  tensor shape) must match training — currently defaulted, needs
  confirmation from the trained model (`model.summary()` / training
  script).
- TUSZ annotation format parsing (`.csv_bi`/`.tse`) is stubbed — needed for
  ROC/confusion matrix to actually populate.
- Free-tier hosting (Render/Railway) may cold-start slowly with TensorFlow
  loaded — acceptable for a demo, worth noting if evaluated live under time
  pressure.
