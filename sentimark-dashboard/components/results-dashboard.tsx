"use client"

import { useEffect, useState } from "react"
import SentimentRing from "@/components/sentiment-ring"
import SentimentVerdict from "@/components/sentiment-verdict"
import ArticlesList from "@/components/articles-list"
import type { PipelineResponse } from "@/types/api"

interface ResultsDashboardProps {
  data: PipelineResponse
}

export default function ResultsDashboard({ data }: ResultsDashboardProps) {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    // Trigger fade-in animation
    const timer = setTimeout(() => setIsVisible(true), 100)
    return () => clearTimeout(timer)
  }, [])

  const { company, sentiment_summary, articles, article_count } = data

  return (
    <div
      className={`mx-auto max-w-[1400px] transition-opacity duration-500 ${isVisible ? "opacity-100" : "opacity-0"}`}
    >
      {/* Company Header */}
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-6">
        <h2 className="font-sans text-3xl font-bold text-white">{company.name}</h2>
        <p className="mt-2 font-mono text-xl font-semibold text-green-500">{company.symbol}</p>
        {company.sector && company.sector !== "Unknown" && (
          <p className="mt-1 font-sans text-sm text-gray-400">{company.sector}</p>
        )}
      </div>

      <div className="mt-8">
        <SentimentVerdict
          sentimentLabel={sentiment_summary.sentiment_label}
          confidence={sentiment_summary.confidence}
        />
      </div>

      <div className="mt-12 grid grid-cols-2 gap-6 lg:grid-cols-4">
        <SentimentRing
          value={(sentiment_summary.bullish / article_count) * 100}
          label={sentiment_summary.bullish.toString()}
          title="Bullish Articles"
          color="green"
        />

        <SentimentRing
          value={(sentiment_summary.neutral / article_count) * 100}
          label={sentiment_summary.neutral.toString()}
          title="Neutral Articles"
          color="yellow"
        />

        <SentimentRing
          value={(sentiment_summary.bearish / article_count) * 100}
          label={sentiment_summary.bearish.toString()}
          title="Bearish Articles"
          color="red"
        />

        <SentimentRing
          value={sentiment_summary.confidence * 100}
          label={`${(sentiment_summary.confidence * 100).toFixed(1)}%`}
          title="Confidence Level"
          color={sentiment_summary.confidence > 0.7 ? "green" : sentiment_summary.confidence > 0.4 ? "yellow" : "red"}
          isPercentage
        />
      </div>

      {/* Articles Section */}
      <div className="mt-12">
        <h3 className="mb-6 font-sans text-2xl font-bold text-white">Analyzed Articles ({article_count} articles)</h3>
        <ArticlesList articles={articles} />
      </div>
    </div>
  )
}
