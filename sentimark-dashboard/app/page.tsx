"use client"

import { useState, useRef } from "react"
import Navbar from "@/components/navbar"
import SearchSection from "@/components/search-section"
import ResultsDashboard from "@/components/results-dashboard"
import Footer from "@/components/footer"
import type { PipelineResponse } from "@/types/api"

export default function Home() {
  const [isLoading, setIsLoading] = useState(false)
  const [isComplete, setIsComplete] = useState(false)
  const [results, setResults] = useState<PipelineResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const resultsRef = useRef<HTMLDivElement>(null)

  const handleSearch = async (symbol: string) => {
    setIsLoading(true)
    setIsComplete(false)
    setError(null)
    setResults(null)

    try {
      const response = await fetch("/api/pipeline", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ symbol }),
      })

      if (!response.ok) {
        throw new Error("Failed to analyze sentiment")
      }

      const data: PipelineResponse = await response.json()
      setResults(data)
      setIsComplete(true)

      // Wait for completion message animation, then scroll
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
      }, 1000)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background Effects Layer */}
      <div className="fixed inset-0 -z-10">
        {/* Radial Gradient Glow */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(16,185,129,0.15)_0%,transparent_70%)]" />

        {/* Grid Pattern Overlay */}
        <div
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage: `
              linear-gradient(rgba(31, 41, 55, 0.3) 1px, transparent 1px),
              linear-gradient(90deg, rgba(31, 41, 55, 0.3) 1px, transparent 1px)
            `,
            backgroundSize: "40px 40px",
          }}
        />

        {/* Animated Mesh Gradient */}
        <div className="absolute inset-0 animate-mesh-gradient opacity-10">
          <div className="absolute inset-0 bg-gradient-to-br from-green-500 via-yellow-400 to-cyan-500 blur-3xl" />
        </div>

        {/* Floating Orbs */}
        <div className="absolute left-1/4 top-1/4 h-96 w-96 animate-float-slow rounded-full bg-green-500/10 blur-3xl" />
        <div className="absolute right-1/4 bottom-1/4 h-80 w-80 animate-float-slower rounded-full bg-cyan-500/10 blur-3xl" />
        <div className="absolute left-1/2 top-1/2 h-72 w-72 animate-float-slowest rounded-full bg-yellow-500/10 blur-3xl" />
      </div>

      <Navbar />

      <main>
        {/* Landing/Hero Section */}
        <section className="flex min-h-screen flex-col items-center justify-center px-4 pt-20">
          <SearchSection onSearch={handleSearch} isLoading={isLoading} isComplete={isComplete} error={error} />
        </section>

        {/* Results Section */}
        {results && (
          <section ref={resultsRef} className="px-4 pb-24">
            <ResultsDashboard data={results} />
          </section>
        )}
      </main>

      <Footer />
    </div>
  )
}
