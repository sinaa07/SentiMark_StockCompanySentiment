"use client"

import { TrendingUp, TrendingDown, Minus } from "lucide-react"

interface SentimentVerdictProps {
  sentimentLabel: string
  confidence: number
}

export default function SentimentVerdict({ sentimentLabel, confidence }: SentimentVerdictProps) {
  const sentimentConfig = {
    bullish: {
      color: "#10b981",
      bgColor: "bg-green-500/10",
      borderColor: "border-green-500/30",
      icon: TrendingUp,
      emoji: "🟢",
      shadowColor: "shadow-green-500/20",
    },
    bearish: {
      color: "#ef4444",
      bgColor: "bg-red-500/10",
      borderColor: "border-red-500/30",
      icon: TrendingDown,
      emoji: "🔴",
      shadowColor: "shadow-red-500/20",
    },
    neutral: {
      color: "#fbbf24",
      bgColor: "bg-yellow-500/10",
      borderColor: "border-yellow-500/30",
      icon: Minus,
      emoji: "🟡",
      shadowColor: "shadow-yellow-500/20",
    },
  }

  const config =
    sentimentConfig[sentimentLabel.toLowerCase() as keyof typeof sentimentConfig] || sentimentConfig.neutral
  const Icon = config.icon

  return (
    <div
      className={`rounded-xl border ${config.borderColor} ${config.bgColor} p-8 shadow-lg ${config.shadowColor} transition-all hover:scale-[1.02]`}
    >
      <div className="flex flex-col items-center gap-6 md:flex-row md:justify-between">
        {/* Sentiment Label */}
        <div className="flex items-center gap-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-gray-900/50">
            <Icon className="h-8 w-8" style={{ color: config.color }} />
          </div>
          <div>
            <p className="font-sans text-sm text-gray-400">Market Sentiment</p>
            <h2 className="font-sans text-4xl font-bold uppercase tracking-wide" style={{ color: config.color }}>
              {sentimentLabel}
            </h2>
          </div>
        </div>

        {/* Confidence Score */}
        <div className="text-center md:text-right">
          <p className="font-sans text-sm text-gray-400">Confidence Score</p>
          <p className="font-mono text-3xl font-bold" style={{ color: config.color }}>
            {(confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>
    </div>
  )
}
