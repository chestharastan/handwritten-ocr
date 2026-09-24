import type { Metadata } from "next";
import { Noto_Sans_Khmer } from "next/font/google";
import "./globals.css";

const khmer = Noto_Sans_Khmer({ subsets: ["khmer"], weight: ["400", "600"], variable: "--font-khmer" });

export const metadata: Metadata = {
  title: "Khmer OCR Tester",
  description: "Crop handwritten Khmer lines from an image and read them with the trained model",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={khmer.variable}>
      <body>{children}</body>
    </html>
  );
}
