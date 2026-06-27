import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Navbar } from "@/components/layout/navbar";
import { ClientLayout } from "@/components/layout/ClientLayout";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "AI Tutor",
  description:
    "Tutor académico personal con IA para la preparación de exámenes universitarios — UTN Santa Fe (CIDISI)",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es" className={inter.variable}>
      <body className="min-h-screen bg-background font-sans text-foreground antialiased">
        <Navbar />
        <ClientLayout>{children}</ClientLayout>
      </body>
    </html>
  );
}
