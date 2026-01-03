import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Navigation } from "@/components/ui/Navigation";

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "Game Arena - LLM Chess Battles",
  description: "Watch AI models compete in blitz chess matches",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${mono.variable} font-sans`}>
        <div className="min-h-screen flex flex-col">
          <Navigation />
          <main className="flex-1 container mx-auto px-4 py-8">
            {children}
          </main>
          <footer className="border-t border-arena-border py-6">
            <div className="container mx-auto px-4 text-center text-sm text-gray-500">
              Game Arena • LLM Chess Battle Platform
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}

