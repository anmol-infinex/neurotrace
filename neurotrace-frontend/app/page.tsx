"use client";

import { useState, useCallback } from "react";
import UploadZone from "@/components/UploadZone";
import PredictionBadge from "@/components/PredictionBadge";
import ProbabilityBars from "@/components/ProbabilityBars";
import WindowCountTable from "@/components/WindowCountTable";
import SustainedEvents from "@/components/SustainedEvents";
import EegChart from "@/components/EegChart";
import MetricsPanel from "@/components/MetricsPanel";
import DownloadButtons from "@/components/DownloadButtons";
import { PredictResult, AppState } from "@/lib/types";
import { runPrediction } from "@/lib/api";

export default function Home() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [edfFile, setEdfFile] = useState<File | null>(null);
  const [annotationFile, setAnnotationFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");

  const handleSubmit = useCallback(async () => {
    if (!edfFile) return;
    setAppState("loading");
    setErrorMsg("");
    setResult(null);

    try {
      const data = await runPrediction(edfFile, annotationFile);
      setResult(data);
      setAppState("result");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error occurred.";
      setErrorMsg(msg);
      setAppState("error");
    }
  }, [edfFile, annotationFile]);

  const handleReset = () => {
    setAppState("idle");
    setResult(null);
    setErrorMsg("");
    setEdfFile(null);
    setAnnotationFile(null);
  };

  return (
    <>
      {/* ─── Header ─── */}
      <header className="header">
        <div className="container header-inner">
          <div className="logo">
            <div className="logo-icon">🧠</div>
            <span className="logo-text">NeuroTrace</span>
          </div>
          <span className="disclaimer-badge">
            ⚠️ Research Prototype — Not for Clinical Use
          </span>
        </div>
      </header>

      {/* ─── Main ─── */}
      <main>
        {/* Hero + Upload (always visible while idle or loading) */}
        {(appState === "idle" || appState === "loading" || appState === "error") && (
          <section className="hero">
            <div className="container">
              <div className="hero-tag">
                <span>🔬</span> TUSZ EEG Analysis Pipeline
              </div>
              <h1 className="hero-title">
                EEG Seizure <span>Detection</span>
              </h1>
              <p className="hero-subtitle">
                Upload a raw EEG recording in <strong>.edf</strong> format.
                The AI model classifies each 10-second window as Normal,
                Pre-Seizure, or Seizure and returns a full clinical report.
              </p>

              {/* Error box */}
              {appState === "error" && (
                <div className="error-box">
                  <span className="error-icon">❌</span>
                  <div className="error-text">
                    <strong>Prediction Failed</strong>
                    <p>{errorMsg}</p>
                  </div>
                </div>
              )}

              <UploadZone
                onEdfFile={setEdfFile}
                onAnnotationFile={setAnnotationFile}
                edfFile={edfFile}
                annotationFile={annotationFile}
                onSubmit={handleSubmit}
                loading={appState === "loading"}
              />
            </div>
          </section>
        )}

        {/* Loading state */}
        {appState === "loading" && (
          <div className="container">
            <div className="loading-wrapper">
              <div className="spinner" />
              <p className="loading-title">Running inference…</p>
              <p className="loading-sub">
                Windowing EDF, running the Keras model, building your report.
                This can take 5–30 seconds depending on file length.
              </p>
            </div>
          </div>
        )}

        {/* ─── Results ─── */}
        {appState === "result" && result && (
          <div className="container" style={{ paddingTop: 24, paddingBottom: 80 }}>
            {/* Back button */}
            <div style={{ marginBottom: 24 }}>
              <button className="btn btn-ghost" onClick={handleReset} id="new-prediction-btn">
                ← New Prediction
              </button>
            </div>

            <div className="results">

              {/* 1 — Prediction badge */}
              <section className="card">
                <PredictionBadge result={result} />
              </section>

              {/* 2 — Two-column: Probabilities + Window Counts */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                <section className="card">
                  <div className="card-header">
                    <span className="card-header-icon">📊</span>
                    <h2 className="card-title">Average Probabilities</h2>
                  </div>
                  <div className="card-body">
                    <ProbabilityBars result={result} />
                  </div>
                </section>

                <section className="card">
                  <div className="card-header">
                    <span className="card-header-icon">🔢</span>
                    <h2 className="card-title">Window Counts</h2>
                  </div>
                  <div className="card-body">
                    <WindowCountTable result={result} />
                  </div>
                </section>
              </div>

              {/* 3 — Sustained events */}
              <section className="card">
                <div className="card-header">
                  <span className="card-header-icon">⚡</span>
                  <h2 className="card-title">Sustained Seizure Events</h2>
                </div>
                <div className="card-body">
                  <SustainedEvents result={result} />
                </div>
              </section>

              {/* 4 — 18-channel EEG trace */}
              <section className="card">
                <div className="card-header">
                  <span className="card-header-icon">📈</span>
                  <h2 className="card-title">18-Channel EEG Signal Preview</h2>
                </div>
                <div className="card-body">
                  <EegChart result={result} />
                </div>
              </section>

              {/* 5 — ROC + Confusion Matrix */}
              <section className="card">
                <div className="card-header">
                  <span className="card-header-icon">🎯</span>
                  <h2 className="card-title">Model Evaluation Metrics</h2>
                </div>
                <div className="card-body">
                  <MetricsPanel result={result} />
                </div>
              </section>

              {/* 6 — Downloads */}
              <section className="card">
                <div className="card-header">
                  <span className="card-header-icon">💾</span>
                  <h2 className="card-title">Download Results</h2>
                </div>
                <div className="card-body">
                  <DownloadButtons urls={result.download_urls} />
                </div>
              </section>

              {/* 7 — Text report */}
              <section className="card">
                <div className="card-header">
                  <span className="card-header-icon">📋</span>
                  <h2 className="card-title">Full Text Report</h2>
                </div>
                <div className="card-body">
                  <pre id="report-text" className="report-pre">
                    {result.report_text}
                  </pre>
                </div>
              </section>

            </div>
          </div>
        )}
      </main>

      {/* ─── Footer ─── */}
      <footer className="footer">
        <div className="container">
          <p>
            NeuroTrace &nbsp;·&nbsp; Research Prototype &nbsp;·&nbsp;
            Not intended for clinical or diagnostic use &nbsp;·&nbsp;
            TUSZ Research Dataset
          </p>
        </div>
      </footer>
    </>
  );
}
