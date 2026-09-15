"use client";

import { PredictResult } from "@/lib/types";

interface PredictionBadgeProps {
  result: PredictResult;
}

const BADGE_CLASS: Record<string, string> = {
  Normal: "badge-normal",
  "Pre-Seizure": "badge-preseizure",
  Seizure: "badge-seizure",
};

const BADGE_EMOJI: Record<string, string> = {
  Normal: "✅",
  "Pre-Seizure": "⚠️",
  Seizure: "🔴",
};

export default function PredictionBadge({ result }: PredictionBadgeProps) {
  const cls = result.final_prediction;
  const badgeClass = BADGE_CLASS[cls] ?? "badge-normal";
  const emoji = BADGE_EMOJI[cls] ?? "🔵";

  return (
    <div className="prediction-hero">
      <p style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 12, letterSpacing: 1, textTransform: "uppercase" }}>
        Final Classification
      </p>
      <div id="prediction-badge" className={`prediction-badge ${badgeClass}`}>
        {emoji} {cls}
      </div>
      <p className="prediction-meta">
        Job ID: {result.job_id} &nbsp;·&nbsp; {result.total_windows} windows analyzed
      </p>
      {result.channel_mismatch_warning && (
        <div className="channel-warning" style={{ marginTop: 20, textAlign: "left" }}>
          <span>⚠️</span>
          <span>{result.channel_mismatch_warning}</span>
        </div>
      )}
    </div>
  );
}
