import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroTrace — EEG Seizure Detection",
  description:
    "Upload a raw EEG .edf recording and receive an automated seizure-risk assessment: classification, probabilities, signal visualization, and clinical report.",
  keywords: ["EEG", "seizure detection", "NeuroTrace", "neurology", "AI", "EDF"],
  openGraph: {
    title: "NeuroTrace — EEG Seizure Detection",
    description: "AI-powered EEG analysis. Upload .edf → get seizure classification report.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <meta name="robots" content="noindex" />
        {/* noindex because this is a research prototype */}
      </head>
      <body>{children}</body>
    </html>
  );
}
