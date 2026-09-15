// NeuroTrace API types — mirrors the FastAPI /predict response shape exactly.

export interface SustainedEvent {
  start: string;   // "HH:MM:SS"
  end: string;     // "HH:MM:SS"
  windows: number;
}

export interface WindowPrediction {
  start_sec: number;
  class: "Normal" | "Pre-Seizure" | "Seizure";
  probabilities: [number, number, number]; // [p_normal, p_preseizure, p_seizure]
}

export interface SignalPreview {
  channels: string[];
  sample_rate_used_for_preview: number;
  data: number[][]; // [channel_idx][sample_idx]
}

export interface DownloadUrls {
  report_txt: string;
  result_json: string;
  roc_png: string | null;
  confusion_png: string | null;
  confusion_txt: string | null;
}

export interface PredictResult {
  job_id: string;
  final_prediction: "Normal" | "Pre-Seizure" | "Seizure";
  average_probabilities: {
    Normal: number;
    "Pre-Seizure": number;
    Seizure: number;
  };
  window_counts: {
    Normal: number;
    "Pre-Seizure": number;
    Seizure: number;
  };
  total_windows: number;
  sustained_events: SustainedEvent[];
  channels_used: string[];
  channel_mismatch_warning: string | null;
  window_predictions: WindowPrediction[];
  signal_preview: SignalPreview;
  has_ground_truth: boolean;
  report_text: string;
  download_urls: DownloadUrls;
}

export type AppState = "idle" | "loading" | "result" | "error";
