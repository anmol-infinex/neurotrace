# NeuroTrace Backend

FastAPI service that takes an uploaded `.edf` file, runs your trained Keras
model, and returns a full prediction report matching your spec doc:
18-channel signal preview, final prediction, average class probabilities,
per-class window counts, sustained seizure events, and (if you supply
ground-truth annotations) ROC curve + confusion matrix as PNG and text.

## 1. Drop in your model

Put your trained model file at:

```
model/seizure_model.keras   (or .h5 — just update MODEL_PATH in app/main.py)
```

## 2. Fix the CONFIG block in `app/main.py`

These four things **must** match how you trained the model, or predictions
will be garbage even though nothing crashes:

- `SAMPLE_RATE` — Hz the model was trained on
- `WINDOW_SEC` / `STEP_SEC` — window length and hop used to build training examples
- `TARGET_CHANNELS` — exact channel montage/order the model expects
- The reshape in `run_inference()` — whether your model wants
  `(batch, samples, channels)` or `(batch, channels, samples)` or something
  else entirely (e.g. spectrogram/feature input instead of raw signal)

## 3. Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Test it:

```bash
curl -F "edf_file=@sample.edf" http://localhost:8000/predict
```

## 4. Deploy (Render — free tier, easiest for this)

1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com) → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Instance type: at least the free/starter tier with 512MB+ RAM (TensorFlow
   needs headroom — if it OOMs, bump the plan).
6. Once deployed you'll get a URL like `https://neurotrace-backend.onrender.com`.

(Railway or Fly.io work the same way if you prefer those.)

## 5. Wire it into the Vercel frontend

In your NeuroTrace Next.js project, replace/extend the local EDF-parsing
logic with a call to this API:

```js
async function runPrediction(edfFile) {
  const formData = new FormData();
  formData.append("edf_file", edfFile);
  // formData.append("annotation_file", annotationFile); // optional, for ROC/confusion matrix

  const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/predict`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error("Prediction failed");
  return res.json(); // { final_prediction, average_probabilities, window_counts,
                      //   sustained_events, signal_preview, download_urls, report_text, ... }
}
```

Add to Vercel project env vars:

```
NEXT_PUBLIC_BACKEND_URL=https://neurotrace-backend.onrender.com
```

Render the 18-channel graph from `result.signal_preview.data` (one array per
channel, already downsampled to ~50Hz so it's light to plot — e.g. with
`recharts` or `chart.js`, one line per channel, offset vertically like a
clinical EEG trace). Render the text report directly from
`result.report_text`. Wire the "download" buttons to
`result.download_urls.*` (they point back at this API and stream the
file/PNG).

## 6. Ground-truth labels (for ROC + confusion matrix)

Those two outputs need **true labels** for the recording, not just
predictions — there's currently a `TODO` in `predict()` where you plug in
your TUSZ annotation parser (`.csv_bi` / `.tse` files map time ranges to
bckg/seiz labels). Without an annotation file, the API still returns
everything else; ROC/confusion endpoints just 404 with a clear message.
