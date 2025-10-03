import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://fintech-forecaster.vercel.app"),
  title: {
    default: "FinTech Forecaster",
    template: "%s | FinTech Forecaster",
  },
  description:
    "Clean and modern dashboard to explore stock, crypto, and forex pricing with Alpha Vantage data.",
  openGraph: {
    title: "FinTech Forecaster",
    description:
      "Clean and modern dashboard to explore stock, crypto, and forex pricing with Alpha Vantage data.",
    url: "https://fintech-forecaster.vercel.app",
    siteName: "FinTech Forecaster",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "FinTech Forecaster dashboard preview",
      },
    ],
  },
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#0f172a" },
    { media: "(prefers-color-scheme: dark)", color: "#f8fafc" },
  ],
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} bg-radial-dusk antialiased text-foreground`}
      >
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <div className="relative flex min-h-screen flex-col">
            <Navbar />
            <main className="flex-1">
              <div className="mx-auto flex w-full max-w-6xl flex-col gap-12 px-6 py-12">
                {children}
              </div>
            </main>
            <Footer />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
