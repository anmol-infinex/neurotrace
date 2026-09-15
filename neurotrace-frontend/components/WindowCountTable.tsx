"use client";

import { PredictResult } from "@/lib/types";

interface WindowCountTableProps {
  result: PredictResult;
}

export default function WindowCountTable({ result }: WindowCountTableProps) {
  const { window_counts, total_windows } = result;

  return (
    <div id="window-count-table" className="win-grid">
      {Object.entries(window_counts).map(([name, count]) => (
        <div key={name} className="win-cell">
          <div className="win-cell-num">{count}</div>
          <div className="win-cell-label">{name}</div>
        </div>
      ))}
      <div className="win-cell total">
        <div className="win-cell-num">{total_windows}</div>
        <div className="win-cell-label">Total</div>
      </div>
    </div>
  );
}
