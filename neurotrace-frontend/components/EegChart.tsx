"use client";

import { PredictResult } from "@/lib/types";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { useMemo } from "react";

interface EegChartProps {
  result: PredictResult;
}

// Clinical EEG stacked trace colors
const CHANNEL_COLORS = [
  "#00d4ff", "#4f7eff", "#a855f7", "#22c55e",
  "#f59e0b", "#ef4444", "#06b6d4", "#8b5cf6",
  "#10b981", "#f97316", "#ec4899", "#6366f1",
  "#14b8a6", "#eab308", "#84cc16", "#3b82f6",
  "#f43f5e", "#0ea5e9",
];

// Vertical offset between channels (in µV equivalent units)
const CHANNEL_OFFSET = 200;

export default function EegChart({ result }: EegChartProps) {
  const { signal_preview } = result;
  const { channels, data, sample_rate_used_for_preview } = signal_preview;

  // Build chart data: one object per sample, with each channel offset vertically
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    const nSamples = data[0].length;
    const maxSamples = Math.min(nSamples, 2000); // cap to 2000 pts for performance
    const step = Math.max(1, Math.floor(nSamples / maxSamples));

    return Array.from({ length: Math.floor(nSamples / step) }, (_, idx) => {
      const sampleIdx = idx * step;
      const timeSec = sampleIdx / sample_rate_used_for_preview;
      const point: Record<string, number> = { time: timeSec };
      channels.forEach((ch, chIdx) => {
        if (data[chIdx]) {
          // Normalize channel amplitude and apply vertical offset
          const raw = data[chIdx][sampleIdx] ?? 0;
          const scale = 1e6; // convert V -> µV
          point[ch] = raw * scale + chIdx * CHANNEL_OFFSET;
        }
      });
      return point;
    });
  }, [data, channels, sample_rate_used_for_preview]);

  const formatTime = (t: number) => {
    const mins = Math.floor(t / 60);
    const secs = Math.floor(t % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  if (chartData.length === 0) {
    return (
      <div style={{ textAlign: "center", padding: 32, color: "var(--text-secondary)" }}>
        No signal preview data available.
      </div>
    );
  }

  return (
    <div id="eeg-chart" className="eeg-chart-wrapper">
      <p style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 12, fontFamily: "JetBrains Mono, monospace" }}>
        {channels.length} channels · ~{sample_rate_used_for_preview} Hz preview · offset {CHANNEL_OFFSET} µV/channel
      </p>
      <ResponsiveContainer width="100%" height={channels.length * 38 + 40}>
        <LineChart data={chartData} margin={{ top: 0, right: 8, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="time"
            tickFormatter={formatTime}
            tick={{ fontSize: 11, fill: "var(--text-muted)", fontFamily: "JetBrains Mono, monospace" }}
            label={{ value: "Time (mm:ss)", position: "insideBottom", offset: -2, fontSize: 11, fill: "var(--text-muted)" }}
            height={30}
          />
          <YAxis
            tick={false}
            tickLine={false}
            axisLine={false}
            width={0}
          />
          <Tooltip
            content={() => null}
          />
          {channels.map((ch, idx) => (
            <Line
              key={ch}
              type="linear"
              dataKey={ch}
              stroke={CHANNEL_COLORS[idx % CHANNEL_COLORS.length]}
              strokeWidth={1}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {/* Channel labels on the right */}
      <div style={{ marginTop: 8, display: "flex", flexWrap: "wrap", gap: "6px 12px" }}>
        {channels.map((ch, idx) => (
          <span
            key={ch}
            style={{
              fontSize: 11,
              color: CHANNEL_COLORS[idx % CHANNEL_COLORS.length],
              fontFamily: "JetBrains Mono, monospace",
            }}
          >
            {ch}
          </span>
        ))}
      </div>
    </div>
  );
}
