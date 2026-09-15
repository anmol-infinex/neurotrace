"use client";

import { DownloadUrls } from "@/lib/types";
import { buildDownloadUrl } from "@/lib/api";

interface DownloadButtonsProps {
  urls: DownloadUrls;
}

interface DownloadItem {
  key: keyof DownloadUrls;
  label: string;
  icon: string;
}

const DOWNLOADS: DownloadItem[] = [
  { key: "report_txt", label: "Text Report", icon: "📄" },
  { key: "result_json", label: "Result JSON", icon: "📦" },
  { key: "roc_png", label: "ROC Curve", icon: "📈" },
  { key: "confusion_png", label: "Confusion Matrix", icon: "🗂️" },
  { key: "confusion_txt", label: "Confusion Text", icon: "📝" },
];

export default function DownloadButtons({ urls }: DownloadButtonsProps) {
  return (
    <div id="download-buttons" className="downloads-grid">
      {DOWNLOADS.map(({ key, label, icon }) => {
        const href = buildDownloadUrl(urls[key] as string | null);
        if (!href) return null;
        return (
          <a
            key={key}
            href={href}
            download
            className="btn btn-download"
            id={`download-${key}`}
          >
            <span>{icon}</span>
            <span>{label}</span>
            <span style={{ marginLeft: 4, opacity: 0.5, fontSize: 12 }}>↓</span>
          </a>
        );
      })}
    </div>
  );
}
