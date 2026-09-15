# design.md — NeuroTrace Detailed Design

## 1. Data Model

### 1.1 Prediction window
```
{
  start_sec: float,
  class: "Normal" | "Pre-Seizure" | "Seizure",
  probabilities: [p_normal, p_pre_seizure, p_seizure]
}
```

### 1.2 Sustained event
```
{ start: "HH:MM:SS", end: "HH:MM:SS", windows: int }
```
A sustained event is a run of `SUSTAINED_MIN_WINDOWS` or more consecutive
windows classified as `Seizure`. This is deliberately simple (run-length
encoding over the window class sequence) rather than a second model — it
turns per-window noise into clinically-readable event ranges, matching the
"Event 1: 00:01:40 -> 00:02:40 | 6 windows" format from the spec.

### 1.3 Job artifact layout
```
outputs/{job_id}/
  {original_filename}.edf
  report.txt
  result.json
  roc_curve.png          (only if ground truth supplied)
  confusion_matrix.png   (only if ground truth supplied)
  confusion_matrix.txt   (only if ground truth supplied)
```

## 2. Processing Pipeline (step-by-step)

1. **Upload** — `.edf` (required) + annotation file (optional) received as
   multipart form data.
2. **Parse** — `mne.io.read_raw_edf(preload=True)`.
3. **Channel alignment** — select the 18 channels the model expects; if
   exact names aren't present, fall back to first-N-available (logged as a
   mismatch — should be tightened before any real evaluation use).
4. **Resample** — to `SAMPLE_RATE` if the file's native rate differs.
5. **Window** — fixed `WINDOW_SEC`/`STEP_SEC` slices across the full
   recording.
6. **Batch inference** — all windows through the model in one `predict()`
   call (fast, avoids per-window Python overhead).
7. **Aggregate** — mean probability per class → average probabilities;
   argmax per window → window counts; argmax of the averages → final
   prediction.
8. **Event detection** — run-length encode the Seizure-window sequence.
9. **(Optional) Evaluation artifacts** — if true labels are available,
   compute ROC (one-vs-rest, per class) and confusion matrix.
10. **Report assembly** — build the exact-format text report, JSON, and
    signal preview payload.
11. **Persist + respond** — write artifacts to the job folder, return JSON
    with inline `report_text` and download URLs.

## 3. Report Format (exact target, from provided spec)

```
============================================================
TUSZ EEG USER EDF PREDICTION REPORT

PREDICTION
------------------------------------------------------------
Final Prediction: Normal

AVERAGE PROBABILITIES
------------------------------------------------------------
Normal: 50.0985%
Pre-Seizure: 17.7766%
Seizure: 32.1249%
------------------------------------------------------------
Normal: 16
Pre-Seizure: 5
Seizure: 9
Total: 30

SUSTAINED SEIZURE EVENTS
------------------------------------------------------------
Event 1: 00:01:40 -> 00:02:40 | 6 windows
```
Implemented verbatim in `build_text_report()`.

## 4. Frontend Design (NeuroTrace UI)

### 4.1 Screens/states
- **Idle** — drag-and-drop zone (existing).
- **Uploading/processing** — spinner/progress while `POST /predict` is
  in-flight.
- **Result view** —
  - Header: final prediction badge (color-coded: green=Normal,
    amber=Pre-Seizure, red=Seizure).
  - 18-channel EEG trace (stacked line chart, one row per channel, using
    `signal_preview.data`).
  - Probability bars (3 classes).
  - Window-count summary table.
  - Sustained-event list/timeline.
  - ROC curve + confusion matrix images, shown only if
    `has_ground_truth` is true; otherwise a note explaining why they're
    absent.
  - Download buttons wired to `download_urls.*`.
- **Error state** — clear message for non-EDF upload, parsing failure, or
  backend unreachable.

### 4.2 18-channel graph rendering approach
- Use `signal_preview.data` (already downsampled server-side to ~50Hz) —
  don't re-fetch or re-parse the raw EDF client-side.
- Render as a stacked multi-line chart with a fixed per-channel vertical
  offset (standard clinical EEG "stacked trace" look), channel labels on
  the left.
- Any chart library that can do multi-series with independent Y-offsets
  (recharts with manual offset, or a canvas-based renderer) works; avoid
  plotting all 18 channels on a shared, unshifted Y-axis — it becomes
  unreadable.

## 5. Error Handling

| Condition | Backend behavior |
|---|---|
| Non-`.edf` upload | `400` with explicit message |
| Recording shorter than one window | `400` with explicit message |
| Model file missing | `500` at first request, explicit path in message |
| Ground truth not supplied | ROC/confusion endpoints return `404` with explanation, rest of response is still complete |
| Channel mismatch | Falls back to first-N channels (should be surfaced as a warning field in the response before production use) |

## 6. Testing Approach

- **Unit:** windowing math (window count for a given duration/step),
  sustained-event run-length logic, report-string formatting against the
  exact spec text.
- **Integration:** one known-good TUSZ `.edf` (+ its annotation file) run
  through `/predict` end-to-end, confirm ROC/confusion matrix generate and
  numbers are sane (probabilities sum to ~1 per window, window counts sum
  to total).
- **Manual/UI:** upload through the live Vercel frontend against the
  deployed backend, confirm graph renders and downloads work.
