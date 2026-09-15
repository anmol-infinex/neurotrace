# AGENTS.md — Execution Instructions for the Coding Agent

You are building **NeuroTrace**: wiring a trained Keras seizure-classification
model into the FastAPI backend in this repo, then deploying backend (Render)
and frontend (Vercel), connected via GitHub.

**Before doing anything else, read in this order:** `PRD.md` →
`architecture.md` → `design.md` → `Phases.md`. They define scope, API
contract, data formats, and phase ordering. Do not deviate from the report
text format in `design.md` §3 — it must match exactly, character for
character.

Repo layout you're starting from:
```
neurotrace-backend/
  app/main.py        <- FastAPI service, already scaffolded and runnable
  model/              <- put the trained model file here
  requirements.txt
  README.md
  PRD.md architecture.md design.md Phases.md AGENTS.md (this file)
```

Frontend repo is separate: the existing Next.js project deployed at
`https://neurotrace-dashboard-eight.vercel.app`. You need its source (ask
the user for the repo/local path if it isn't already in your workspace —
do not assume its structure, inspect it first).

Work through the steps below **in order**. Do not skip ahead to deployment
before Phase 1–4 are locally verified. After each numbered step, verify the
listed acceptance criteria before continuing.

---

## Step 1 — Model integration (blocks everything else)

1. Confirm the model file is present at `model/seizure_model.keras` (or
   `.h5` — update `MODEL_PATH` in `app/main.py` if so).
2. Inspect it to get ground truth on preprocessing:
   ```python
   import tensorflow as tf
   m = tf.keras.models.load_model("model/seizure_model.keras")
   m.summary()
   print(m.input_shape)
   ```
3. Update the `CONFIG` block at the top of `app/main.py`:
   `SAMPLE_RATE`, `WINDOW_SEC`, `STEP_SEC`, `TARGET_CHANNELS`, and the
   `class` order in `CLASS_NAMES` to match training exactly.
4. Fix the `np.transpose` reshape in `run_inference()` if the model's
   `input_shape` isn't `(None, samples, channels)`.

**Acceptance:** `m.input_shape` matches what `run_inference()` feeds it —
no shape-mismatch error when you run Step 3 below.

## Step 2 — Ground-truth annotation parsing (for ROC + confusion matrix)

1. Get one sample TUSZ `.edf` + its matching annotation file from the user.
2. Identify the annotation format (`.csv_bi`, `.tse`, etc.) by inspecting it.
3. Implement `parse_tusz_annotations(annotation_path, starts_sec, WINDOW_SEC)`
   in `app/main.py`, returning a true-class-index array aligned to
   `starts_sec`.
4. Wire it into the `if annotation_file is not None:` branch in `predict()`.

**Acceptance:** posting the sample file + annotation to `/predict` returns
`has_ground_truth: true` and non-404 `roc.png`/`confusion.png` URLs.

## Step 3 — Local backend verification

```bash
cd neurotrace-backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# separate terminal:
curl -F "edf_file=@/path/to/sample.edf" http://localhost:8000/predict | python -m json.tool
```

**Acceptance:** valid JSON returned, `average_probabilities` values sum to
~100, `window_counts` values sum to `total_windows`, `report_text` matches
the format in `design.md` §3 exactly.

## Step 4 — Frontend integration

Work in the **existing** NeuroTrace Next.js repo, not this one.

1. Locate where the current client-side EDF parsing/"Local EEG engine"
   happens.
2. Add an API call per `architecture.md` §3 / the snippet in
   `neurotrace-backend/README.md` §5 — `POST` the file to
   `${NEXT_PUBLIC_BACKEND_URL}/predict`.
3. Build the result view per `design.md` §4: prediction badge, probability
   bars, window-count table, sustained-event list, 18-channel stacked trace
   from `signal_preview.data`, conditional ROC/confusion-matrix images,
   download buttons from `download_urls.*`.
4. Add `.env.local`:
   ```
   NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
   ```
5. Run the frontend locally against the local backend from Step 3, upload a
   real file, confirm the full flow renders.

**Acceptance:** local frontend → local backend round trip works end-to-end
in the browser, including all download buttons.

## Step 5 — GitHub

Do this for **both** repos (backend and frontend) if either isn't already
on GitHub.

```bash
cd neurotrace-backend
git init                                   # skip if already a repo
git add .
git commit -m "NeuroTrace backend: EDF inference API"
gh repo create neurotrace-backend --private --source=. --remote=origin --push
```
If `gh` (GitHub CLI) isn't authenticated, run `gh auth login` first, or
create the repo manually on github.com and:
```bash
git remote add origin https://github.com/<user>/neurotrace-backend.git
git branch -M main
git push -u origin main
```

**Do not commit the trained model file if it's large** — check its size
first (`ls -lh model/`). If it's over ~50MB, add `model/*.keras` /
`model/*.h5` to `.gitignore` and instead upload the model directly in the
Render dashboard or via a release asset / cloud storage link, per Render's
docs. If it's small, commit it normally.

**Acceptance:** `git status` clean, remote `main` branch has the latest
commit, visible on github.com.

## Step 6 — Backend deployment (Render)

1. On [render.com](https://render.com): New → Web Service → connect the
   `neurotrace-backend` GitHub repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Instance size: at least the tier with 512MB+ RAM (TensorFlow needs
   headroom; bump if it OOMs on first request).
5. If the model file wasn't committed to git, upload it into the service's
   persistent disk / attach it per Render's file-upload or disk docs, at
   the path `model/seizure_model.keras` relative to the app root.
6. Deploy, wait for build to finish, note the resulting URL (e.g.
   `https://neurotrace-backend.onrender.com`).
7. Hit `GET https://<that-url>/health` — confirm `{"status":"ok","model_loaded":true}`.

**Acceptance:** `/health` returns `model_loaded: true` on the live URL.

## Step 7 — Frontend deployment (Vercel)

The frontend project is already on Vercel at
`neurotrace-dashboard-eight.vercel.app`. Update it rather than recreating it:

1. Push the Step 4 changes to the frontend's GitHub repo (same pattern as
   Step 5) — if it's already connected to Vercel via GitHub, this alone
   triggers a redeploy.
2. In the Vercel project dashboard → Settings → Environment Variables, set:
   ```
   NEXT_PUBLIC_BACKEND_URL=https://neurotrace-backend.onrender.com
   ```
   (use the real Step 6 URL). Apply to Production (and Preview if you want
   preview deployments to also hit the live backend).
3. Redeploy (Vercel does this automatically on push if GitHub-connected;
   otherwise trigger manually from the dashboard).

**Acceptance:** the live Vercel URL, when uploading a `.edf`, successfully
calls the Render backend and renders real results (check the browser
Network tab for a `200` from `/predict`).

## Step 8 — Lock down CORS

In `neurotrace-backend/app/main.py`, confirm `allow_origins` in the
`CORSMiddleware` config lists the **exact** production Vercel URL (already
present: `https://neurotrace-dashboard-eight.vercel.app`) — remove any
wildcard or extra dev origins before calling this done, then redeploy the
backend if you changed it.

**Acceptance:** cross-origin request from the production frontend succeeds;
a request from a random other origin is blocked (verify with browser dev
tools or curl `-H "Origin: https://evil.example.com"`).

## Step 9 — Final end-to-end smoke test

From the live Vercel URL: upload a real `.edf` (with annotation file, if
your UI supports attaching one) and confirm:
- [ ] Prediction badge shows correct final class
- [ ] Probability bars match `average_probabilities`
- [ ] 18-channel graph renders
- [ ] Sustained seizure events list correctly (if any in the file)
- [ ] ROC + confusion matrix render (if annotation was supplied)
- [ ] All download buttons produce the correct file

Report back to the user with the live URLs (frontend + backend `/health`)
and flag anything from `Phases.md` Phase 6 (polish) you didn't get to.

---

## Rules for the agent

- Do not invent values for `SAMPLE_RATE`, `WINDOW_SEC`, `TARGET_CHANNELS`,
  or the model's input shape — get them from the actual model file or ask
  the user. Wrong values fail silently (no crash, just wrong predictions).
- Do not change the report text format in `design.md` §3.
- Do not commit secrets, API keys, or (if large/sensitive) raw patient-style
  EEG files to git.
- If blocked (missing model file, missing frontend repo access, missing
  GitHub/Vercel/Render credentials), stop and ask the user rather than
  guessing or fabricating placeholder deployments.
