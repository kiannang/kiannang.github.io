import type { Metadata } from "next";
import { EB_Garamond } from "next/font/google";
import "./globals.css";

const garamond = EB_Garamond({
  subsets: ["latin"],
  variable: "--font-garamond",
  weight: ["400", "500"],
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
      <body className={`${garamond.variable} pinstripe font-sans text-foreground antialiased`}>
        {children}
      </body>
    </html>
  );
}
