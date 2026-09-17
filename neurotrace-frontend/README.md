# 🧠 NeuroTrace Frontend — Clinical EEG Telemetry Dashboard

[![Vercel Live Deployment](https://img.shields.io/badge/Vercel-neurotrace--dashboard-00d4ff?style=for-the-badge&logo=vercel&logoColor=white)](https://neurotrace-dashboard-eight.vercel.app)
[![Next.js 16](https://img.shields.io/badge/Next.js-16.3.5-black?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![React 19](https://img.shields.io/badge/React-19.2.8-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![Recharts](https://img.shields.io/badge/Recharts-3.10.1-22c55e?style=for-the-badge)](https://recharts.org/)

Next.js 16 web application delivering real-time electroencephalogram (EEG) visualization, tri-state seizure prediction diagnostics, and clinical telemetry analytics.

🔗 **Live Testing Application**: [https://neurotrace-dashboard-eight.vercel.app](https://neurotrace-dashboard-eight.vercel.app)  
📡 **Backend Inference API (Render)**: [https://neurotrace-4fpu.onrender.com](https://neurotrace-4fpu.onrender.com)

---

## 🌟 Dashboard Features

- ⚡ **1-Click Instant Test Mode**: Includes a pre-bundled, clinical-grade sample (`tusz_verified_sample.edf`) in `/public` allowing instant zero-configuration testing.
- 🌊 **Clinical 18-Channel Stacked Trace Visualizer**: High-performance SVG/Canvas rendering of multi-channel differential EEG waveforms with calibrated microvolt ($200\,\mu\text{V}$) vertical spacing.
- 📊 **Per-Window Prediction Timeline & Probability Bars**: Visual breakdown of Normal (Background), Pre-Seizure (Preictal), and Seizure (Ictal) classifications.
- ⏱️ **Sustained Seizure Event Detection**: Run-length temporal filtering alerting clinicians to continuous epileptiform activity ($\ge 2$ consecutive 10-second windows).
- 📄 **Diagnostic Artifact Exporter**: One-click downloads for diagnostic reports (`.txt`), raw machine-readable JSON telemetry, ROC curves, and confusion matrices.
- 🌌 **Clinical Glassmorphism Aesthetics**: Built with customized CSS tokens, dark clinical palette, and fluid typography.

---

## 🚀 Quickstart & Local Development

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment Variables
Create a `.env.local` file in this directory:
```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```
> Note: For production, point `NEXT_PUBLIC_BACKEND_URL` to your live FastAPI service (e.g. `https://neurotrace-4fpu.onrender.com`).

### 3. Run Local Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the application.

---

## 📦 Project Structure

```
neurotrace-frontend/
├── app/
│   ├── favicon.ico
│   ├── globals.css         # Clinical dark mode design tokens & glassmorphism
│   ├── layout.tsx          # Root layout & font loading (Inter, JetBrains Mono)
│   ├── page.module.css
│   └── page.tsx            # Main dashboard controller & state orchestration
├── components/
│   ├── DownloadButtons.tsx # Report & image download handlers
│   ├── EegChart.tsx        # 18-channel Recharts clinical waveform renderer
│   ├── MetricsPanel.tsx    # Summary cards & class distribution
│   ├── PredictionBadge.tsx # High-contrast tri-state status badge
│   ├── ProbabilityBars.tsx # Animated class probability meters
│   ├── SustainedEvents.tsx # Temporal sustained seizure alert log
│   ├── UploadZone.tsx      # Drag-and-drop EDF uploader + 1-Click sample test
│   └── WindowCountTable.tsx# Tabular window counts & percentages
├── lib/
│   ├── api.ts              # FastAPI client & FormData serialization
│   └── types.ts            # TypeScript interfaces & state models
└── public/
    └── sample.edf          # Pre-loaded verified TUSZ sample EEG file
```

---

## 🚢 Deploying to Vercel

1. Push your changes to GitHub.
2. Import the repository into [Vercel](https://vercel.com).
3. Set **Root Directory** to `neurotrace-frontend`.
4. Add the environment variable:
   - `NEXT_PUBLIC_BACKEND_URL`: `https://neurotrace-4fpu.onrender.com`
5. Deploy.
