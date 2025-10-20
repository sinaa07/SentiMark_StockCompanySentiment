import type React from "react"
import type { Metadata } from "next"
import localFont from "next/font/local"
import "./globals.css"

const inter = localFont({
  src: "../public/fonts/Inter-Regular.woff2",
  variable: "--font-inter",
  display: "swap"
})

const robotoMono = localFont({
  src: "../public/fonts/RobotoMono-Regular.woff2",
  variable: "--font-roboto-mono",
  display: "swap"
})

export const metadata: Metadata = {
  title: "SentiMark - NSE Market Sentiment Dashboard",
  description: "Real-time sentiment analysis for NSE stocks",
  generator: "v0.app"
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${robotoMono.variable}`}>
      <body className="bg-black text-white antialiased">{children}</body>
    </html>
  )
}