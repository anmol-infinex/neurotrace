# architecture.md — NeuroTrace System Architecture

## 1. High-Level Diagram

```
┌─────────────────────────┐         HTTPS / multipart POST        ┌──────────────────────────────┐
│   Frontend (Vercel)     │  ───────────────────────────────────► │   Backend (Render/Railway)   │
│   Next.js — NeuroTrace  │                                        │   FastAPI + TensorFlow/Keras │
│   neurotrace-dashboard  │  ◄─────────────────────────────────── │   app/main.py                │
└─────────────────────────┘         JSON + file downloads          └──────────────────────────────┘
                                                                              │
                                                                              ▼
                                                                    ┌──────────────────┐
                                                                    │  Trained model    │
                                                                    │  seizure_model    │
                                                                    │  .keras (18-ch)   │
                                                                    └──────────────────┘
                                                                              │
                                                                              ▼
                                                                    ┌──────────────────┐
                                                                    │  Per-job outputs  │
                                                                    │  /outputs/{uuid}/ │
                                                                    │  report, PNGs,    │
                                                                    │  JSON             │
                                                                    └──────────────────┘
```

## 2. Components

### 2.1 Frontend — `neurotrace-dashboard` (Vercel, Next.js)
- Drag-and-drop `.edf` upload UI (already built).
- Calls backend `POST /predict` with the file as `multipart/form-data`.
- Renders: 18-channel EEG trace, prediction report card, sustained-event
  timeline, ROC/confusion-matrix images (when present), download buttons.
- Reads backend URL from `NEXT_PUBLIC_BACKEND_URL` env var.

### 2.2 Backend — FastAPI service (Render/Railway)
- Single service, `app/main.py`.
- Loads the Keras model once at startup (module-level singleton — see
  `get_model()`), not per-request, to avoid reload latency.
- Stateless per request except for writing job artifacts to
  `outputs/{job_id}/` on local disk (ephemeral — fine for a demo; would move
  to object storage like S3 for a production version).
- CORS restricted to the known frontend origin(s).

### 2.3 EDF Processing Pipeline (inside the backend)
1. `mne.io.read_raw_edf` — load and parse the file.
2. Channel selection/renaming to the 18-channel target montage.
3. Resample to the model's trained sample rate.
4. Windowing: fixed-length, non-overlapping windows (`WINDOW_SEC`,
   `STEP_SEC`).
5. Reshape to the model's expected input tensor shape.

### 2.4 Model Layer
- Keras `.keras`/`.h5` model, loaded via `tf.keras.models.load_model`.
- Input: one window at a time (batched) → output: per-class probabilities
  for `["Normal", "Pre-Seizure", "Seizure"]`.
- Swappable: model file lives in `model/`, referenced by `MODEL_PATH` —
  replacing the file (with matching input shape) requires no code changes
  elsewhere except the `CONFIG` block if preprocessing changes.

### 2.5 Report/Artifact Generation
- Aggregation: mean probability per class, per-class window counts, argmax
  for final prediction.
- Sustained-event detection: run-length encoding over consecutive
  Seizure-classified windows, thresholded by `SUSTAINED_MIN_WINDOWS`.
- Text report: string-built to match the exact spec format.
- ROC curve (`sklearn.metrics.roc_curve`, one-vs-rest per class) and
  confusion matrix (`sklearn.metrics.confusion_matrix`) — only generated
  when ground-truth labels are available (annotation upload).
- All artifacts written to a UUID-scoped job directory and served back via
  dedicated `GET` endpoints.

## 3. API Contract

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + model-loaded check |
| `POST` | `/predict` | Upload `.edf` (+ optional annotation file) → full prediction JSON |
| `GET` | `/predict/{job_id}/report` | Download text report |
| `GET` | `/predict/{job_id}/json` | Download raw JSON result |
| `GET` | `/predict/{job_id}/roc.png` | ROC curve image (404 if no ground truth) |
| `GET` | `/predict/{job_id}/confusion.png` | Confusion matrix image (404 if no ground truth) |
| `GET` | `/predict/{job_id}/confusion.txt` | Confusion matrix as text (404 if no ground truth) |

`POST /predict` response shape:

```json
{
  "job_id": "uuid",
  "final_prediction": "Normal",
  "average_probabilities": {"Normal": 50.09, "Pre-Seizure": 17.78, "Seizure": 32.12},
  "window_counts": {"Normal": 16, "Pre-Seizure": 5, "Seizure": 9},
  "total_windows": 30,
  "sustained_events": [{"start": "0:01:40", "end": "0:02:40", "windows": 6}],
  "channels_used": ["FP1-F7", "..."],
  "window_predictions": [{"start_sec": 0, "class": "Normal", "probabilities": [0.7,0.2,0.1]}],
  "signal_preview": {"channels": ["..."], "sample_rate_used_for_preview": 50, "data": [[...], [...]]},
  "has_ground_truth": false,
  "report_text": "full text report",
  "download_urls": {"report_txt": "...", "result_json": "...", "roc_png": null, "confusion_png": null, "confusion_txt": null}
}
```

## 4. Deployment Topology

- **Frontend:** Vercel (already deployed) — static/SSR Next.js.
- **Backend:** Render or Railway, one always-on (or on-demand) web service
  running `uvicorn app.main:app`.
- **Model artifact:** bundled into the backend repo/image (`model/`
  directory) — fine at typical Keras model sizes; move to object storage
  if the model grows large.
- **CORS:** backend allow-list contains the exact Vercel production URL
  (and `localhost:3000` for local frontend dev).

## 5. Security Considerations

- Validate file extension and, ideally, magic bytes before parsing (avoid
  passing arbitrary uploaded files straight into a parser).
- Set a max upload size at the FastAPI/reverse-proxy level to prevent
  large-file DoS.
- Job directories are UUID-named and not enumerable, but are not
  access-controlled — acceptable for a public demo of non-sensitive
  research data (TUSZ), not for real patient data.
- Pin dependency versions (`requirements.txt`) to avoid supply-chain drift
  in a public-facing service.

## 6. Extensibility Notes

- Swapping the model: replace the file in `model/`, adjust `CONFIG`
  (sample rate, window size, channel list, input reshape) to match the new
  model's training setup.
- Adding real-time/streaming input later (per the original Seizure-Sentry
  concept) would mean adding a WebSocket or chunked-ingestion endpoint
  alongside this batch `/predict` endpoint — the windowing/inference/report
  logic here is reusable as-is.
