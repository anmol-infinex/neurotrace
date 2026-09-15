import { PredictResult } from "./types";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

/**
 * POST /predict — sends the .edf file (and optional annotation) to the backend.
 * Returns the full PredictResult JSON.
 */
export async function runPrediction(
  edfFile: File,
  annotationFile?: File | null
): Promise<PredictResult> {
  const formData = new FormData();
  formData.append("edf_file", edfFile);
  if (annotationFile) {
    formData.append("annotation_file", annotationFile);
  }

  const res = await fetch(`${BACKEND_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let detail = `Server error (${res.status})`;
    try {
      const err = await res.json();
      if (err.detail) detail = err.detail;
    } catch {
      /* ignore JSON parse error */
    }
    throw new Error(detail);
  }

  return res.json() as Promise<PredictResult>;
}

/**
 * Build an absolute download URL for a job artifact.
 * The paths returned by the backend are relative (e.g. /predict/{id}/report).
 */
export function buildDownloadUrl(relativePath: string | null): string | null {
  if (!relativePath) return null;
  return `${BACKEND_URL}${relativePath}`;
}

/**
 * GET /health — lightweight check that the backend is reachable.
 */
export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  const res = await fetch(`${BACKEND_URL}/health`);
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json();
}
