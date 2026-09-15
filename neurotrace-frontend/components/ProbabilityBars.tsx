"use client";

import { PredictResult } from "@/lib/types";

interface ProbabilityBarsProps {
  result: PredictResult;
}

const COLOR_MAP: Record<string, { fill: string; textColor: string }> = {
  Normal: { fill: "fill-green", textColor: "var(--green)" },
  "Pre-Seizure": { fill: "fill-amber", textColor: "var(--amber)" },
  Seizure: { fill: "fill-red", textColor: "var(--red)" },
};

export default function ProbabilityBars({ result }: ProbabilityBarsProps) {
  const entries = Object.entries(result.average_probabilities) as [string, number][];

  return (
    <div className="prob-list">
      {entries.map(([name, pct]) => {
        const { fill, textColor } = COLOR_MAP[name] ?? { fill: "fill-green", textColor: "var(--green)" };
        return (
          <div key={name} id={`prob-bar-${name.replace(/[^a-z]/gi, "").toLowerCase()}`}>
            <div className="prob-item-label">
              <span className="prob-name">{name}</span>
              <span className="prob-pct" style={{ color: textColor }}>
                {pct.toFixed(2)}%
              </span>
            </div>
            <div className="prob-bar-bg">
              <div
                className={`prob-bar-fill ${fill}`}
                style={{ width: `${Math.min(pct, 100)}%` }}
                role="progressbar"
                aria-valuenow={pct}
                aria-valuemin={0}
                aria-valuemax={100}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
