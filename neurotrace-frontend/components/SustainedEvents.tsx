"use client";

import { PredictResult } from "@/lib/types";

interface SustainedEventsProps {
  result: PredictResult;
}

export default function SustainedEvents({ result }: SustainedEventsProps) {
  const { sustained_events } = result;

  if (sustained_events.length === 0) {
    return (
      <div className="no-events">
        <strong>✅ No Sustained Seizure Events Detected</strong>
        No consecutive sequences of {3}+ seizure-classified windows were found.
      </div>
    );
  }

  return (
    <div id="sustained-events-list" className="event-list">
      {sustained_events.map((ev, i) => (
        <div key={i} className="event-item">
          <span className="event-num">EVENT {i + 1}</span>
          <span className="event-time">{ev.start}</span>
          <span className="event-sep">→</span>
          <span className="event-time">{ev.end}</span>
          <span className="event-wins">{ev.windows} windows</span>
        </div>
      ))}
    </div>
  );
}
