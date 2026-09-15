# Phases.md — NeuroTrace Build Plan

## Phase 0 — Backend Scaffold ✅ (done)
- FastAPI project structure, `/predict` + download endpoints
- EDF parsing, windowing, batch inference, aggregation, text report builder
- ROC/confusion-matrix generation logic (pending ground-truth wiring)
- `requirements.txt`, deployment README

## Phase 1 — Model Integration (do this first, blocks everything else)
- [ ] Drop trained model into `model/seizure_model.keras`
- [ ] Confirm from `model.summary()` (or training script): input shape,
      sample rate, window length, channel order/count, class order
- [ ] Update `CONFIG` block in `app/main.py` to match exactly
- [ ] Fix the reshape in `run_inference()` if the model expects something
      other than `(batch, samples, channels)`
- [ ] Run one real TUSZ `.edf` through `/predict` locally, sanity-check the
      output numbers (probabilities sum to 1, window counts sum to total)

## Phase 2 — Ground-Truth Labels (for ROC + confusion matrix)
- [ ] Identify the TUSZ annotation file format you have (`.csv_bi`, `.tse`,
      or similar) for the files you'll demo with
- [ ] Implement `parse_tusz_annotations()` — map each window's start time to
      a true class index
- [ ] Wire it into the `annotation_file` branch in `predict()`
- [ ] Confirm ROC/confusion matrix PNGs generate correctly on a labeled file

## Phase 3 — Local End-to-End Test
- [ ] Run backend locally (`uvicorn app.main:app --reload`)
- [ ] Point the Vercel frontend at `http://localhost:8000` via
      `NEXT_PUBLIC_BACKEND_URL` (local `.env`) for dev testing
- [ ] Upload a file through the real UI, confirm the full flow: upload →
      predict → report renders → graph renders → downloads work

## Phase 4 — Frontend Wiring (production)
- [ ] Replace/extend the existing client-side EDF logic to call
      `POST /predict` on the backend
- [ ] Build the result view: prediction badge, probability bars, window
      counts, sustained-event list, 18-channel stacked trace
- [ ] Conditionally render ROC/confusion-matrix sections based on
      `has_ground_truth`
- [ ] Wire download buttons to `download_urls.*`
- [ ] Handle loading and error states

## Phase 5 — Deployment
- [ ] Push backend to GitHub, deploy on Render (or Railway)
- [ ] Set `NEXT_PUBLIC_BACKEND_URL` in Vercel project settings to the
      deployed backend URL
- [ ] Update backend CORS `allow_origins` to the exact production Vercel URL
- [ ] Smoke-test the live site end-to-end

## Phase 6 — Polish / Demo-Readiness
- [ ] Add a max-upload-size limit and friendly error for oversized files
- [ ] Add a short loading/progress indicator (TensorFlow inference isn't
      instant)
- [ ] Prepare 1–2 known-good sample `.edf` files (with annotations) to use
      live during grading/demo so ROC + confusion matrix are guaranteed to
      populate
- [ ] Final pass on the "prototype / not for clinical use" messaging and
      overall UI polish
- [ ] Record a short backup demo video/GIF in case live inference is slow
      or the free-tier backend cold-starts during grading

## Suggested ordering under time pressure
If time is short: **Phase 1 → Phase 3 → Phase 4 → Phase 5** are the
critical path to a working, demoable product. Phase 2 (ROC/confusion
matrix) and Phase 6 (polish) can be dropped from the live demo if needed —
the core prediction report and 18-channel graph alone already satisfy most
of the spec.
