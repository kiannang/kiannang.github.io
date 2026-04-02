import type { Metadata } from "next";
import { IBM_Plex_Mono, Manrope, Playfair_Display } from "next/font/google";
import "./globals.css";

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-manrope",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500"],
});

const playfair = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["400", "700"],
  style: ["normal", "italic"],
});

export const metadata: Metadata = {
  title: "Kianna Ng | Research Portfolio",
  description:
    "Portfolio for Kianna Ng, a PhD student in Electrical Engineering and Computer Science at UC Merced working across human-centered AI, multimodal learning, robotics, and intelligent systems.",
  openGraph: {
    title: "Kianna Ng | Research Portfolio",
    description:
      "A research-forward portfolio centered on human-centered AI, multimodal learning, robotics, and intelligent systems.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Kianna Ng | Research Portfolio",
    description:
      "A research-forward portfolio centered on human-centered AI, multimodal learning, robotics, and intelligent systems.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${manrope.variable} ${mono.variable} ${playfair.variable} bg-background font-sans text-foreground antialiased`}>
        {children}
      </body>
    </html>
  );
}
