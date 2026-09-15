"use client";

import { useCallback, useRef, useState } from "react";

interface UploadZoneProps {
  onEdfFile: (file: File) => void;
  onAnnotationFile: (file: File | null) => void;
  edfFile: File | null;
  annotationFile: File | null;
  onSubmit: () => void;
  loading: boolean;
}

export default function UploadZone({
  onEdfFile,
  onAnnotationFile,
  edfFile,
  annotationFile,
  onSubmit,
  loading,
}: UploadZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const annotInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      const file = Array.from(e.dataTransfer.files).find((f) =>
        f.name.toLowerCase().endsWith(".edf")
      );
      if (file) onEdfFile(file);
    },
    [onEdfFile]
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onEdfFile(file);
  };

  const handleAnnotChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onAnnotationFile(e.target.files?.[0] ?? null);
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  return (
    <div style={{ maxWidth: 600, margin: "0 auto" }}>
      {/* Drag-and-drop zone */}
      <div
        id="edf-upload-zone"
        className={`upload-zone${dragOver ? " drag-over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload EDF file"
        onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
      >
        <div className="upload-icon">🧠</div>

        {edfFile ? (
          <>
            <div className="file-pill">
              <span>📄</span>
              <span>{edfFile.name}</span>
              <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>
                ({formatSize(edfFile.size)})
              </span>
            </div>
            <p className="upload-sub">
              Click or drop to change file
            </p>
          </>
        ) : (
          <>
            <p className="upload-title">Drop your EEG recording here</p>
            <p className="upload-sub">
              Drag & drop a <span>.edf</span> file, or click to browse
            </p>
          </>
        )}

        <input
          ref={fileInputRef}
          id="edf-file-input"
          type="file"
          accept=".edf"
          style={{ display: "none" }}
          onChange={handleFileChange}
        />
      </div>

      {/* 1-Click Sample File Option */}
      <div style={{ textAlign: "center", marginTop: 10, marginBottom: 16 }}>
        <button
          type="button"
          onClick={async (e) => {
            e.stopPropagation();
            try {
              const res = await fetch("/sample.edf");
              const blob = await res.blob();
              const file = new File([blob], "tusz_verified_sample.edf", { type: "application/octet-stream" });
              onEdfFile(file);
            } catch (err) {
              console.error("Failed to load sample:", err);
            }
          }}
          className="btn btn-ghost"
          style={{ fontSize: 13, color: "#38bdf8", cursor: "pointer" }}
        >
          ✨ Try with pre-loaded Sample EEG File (1-Click)
        </button>
      </div>

      {/* Annotation file (optional) */}
      <div className="annotation-row">
        <span>🏷️</span>
        <span>Ground-truth annotation (optional — enables ROC + confusion matrix):</span>
        <label
          htmlFor="annotation-file-input"
          className="annotation-label"
        >
          {annotationFile ? annotationFile.name : "Attach .csv_bi / .tse"}
        </label>
        <input
          ref={annotInputRef}
          id="annotation-file-input"
          type="file"
          accept=".csv_bi,.tse,.csv"
          onChange={handleAnnotChange}
        />
        {annotationFile && (
          <button
            className="btn btn-ghost"
            style={{ padding: "3px 8px", fontSize: 11 }}
            onClick={(e) => {
              e.stopPropagation();
              onAnnotationFile(null);
              if (annotInputRef.current) annotInputRef.current.value = "";
            }}
          >
            ✕
          </button>
        )}
      </div>

      {/* Submit */}
      <div style={{ textAlign: "center", marginTop: 28 }}>
        <button
          id="run-prediction-btn"
          className="btn btn-primary"
          disabled={!edfFile || loading}
          onClick={onSubmit}
          style={{ fontSize: 16, padding: "14px 36px" }}
        >
          {loading ? (
            <>
              <span
                style={{
                  width: 16,
                  height: 16,
                  border: "2px solid rgba(0,0,0,0.3)",
                  borderTopColor: "#000",
                  borderRadius: "50%",
                  display: "inline-block",
                  animation: "spin 0.8s linear infinite",
                }}
              />
              Running inference…
            </>
          ) : (
            <>⚡ Run Prediction</>
          )}
        </button>
      </div>
    </div>
  );
}
