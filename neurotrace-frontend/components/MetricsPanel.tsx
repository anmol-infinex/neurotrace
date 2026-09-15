"use client";

import { PredictResult } from "@/lib/types";
import { buildDownloadUrl } from "@/lib/api";

interface MetricsPanelProps {
  result: PredictResult;
}

export default function MetricsPanel({ result }: MetricsPanelProps) {
  const backendUrl =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

  if (!result.has_ground_truth) {
    return (
      <div className="no-ground-truth">
        <strong>📊 ROC Curve & Confusion Matrix</strong>
        These evaluation metrics require a ground-truth annotation file (
        <code>.csv_bi</code> or <code>.tse</code>) to be uploaded alongside
        the EDF. Re-upload with an annotation file to enable them.
      </div>
    );
  }

  const rocUrl = buildDownloadUrl(result.download_urls.roc_png);
  const confUrl = buildDownloadUrl(result.download_urls.confusion_png);

  return (
    <div id="metrics-panel" className="metrics-grid">
      {rocUrl && (
        <div className="metric-img-card">
          <div className="metric-img-label">ROC Curve (One-vs-Rest)</div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={rocUrl} alt="ROC Curve" loading="lazy" />
        </div>
      )}
      {confUrl && (
        <div className="metric-img-card">
          <div className="metric-img-label">Confusion Matrix</div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={confUrl} alt="Confusion Matrix" loading="lazy" />
        </div>
      )}
    </div>
  );
}
