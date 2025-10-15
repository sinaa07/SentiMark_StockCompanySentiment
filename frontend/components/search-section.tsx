"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { Search, TrendingUp, Zap, Brain, CheckCircle } from "lucide-react"
import { Input } from "@/components/ui/input"
import SegmentedRingLoader from "@/components/segmented-ring-loader"
import RecentSearches from "@/components/recent-searches"
import type { AutocompleteResult } from "@/types/api"

interface SearchSectionProps {
  onSearch: (symbol: string) => void
  isLoading: boolean
  isComplete: boolean
  error: string | null
}

export default function SearchSection({ onSearch, isLoading, isComplete, error }: SearchSectionProps) {
  const [query, setQuery] = useState("")
  const [suggestions, setSuggestions] = useState<AutocompleteResult[]>([])
  const [showDropdown, setShowDropdown] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(-1)
  const debounceTimer = useRef<NodeJS.Timeout>()

  useEffect(() => {
    if (query.length < 2) {
      setSuggestions([])
      setShowDropdown(false)
      return
    }

    // Debounce search
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current)
    }

    debounceTimer.current = setTimeout(async () => {
      try {
        const response = await fetch(`/api/autocomplete?query=${encodeURIComponent(query)}`)
        if (response.ok) {
          const data: AutocompleteResult[] = await response.json()
          setSuggestions(data)
          setShowDropdown(data.length > 0)
        }
      } catch (err) {
        console.error("[v0] Autocomplete error:", err)
      }
    }, 300)

    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current)
      }
    }
  }, [query])

  const handleSelect = (symbol: string) => {
    setQuery("")
    setSuggestions([])
    setShowDropdown(false)
    setSelectedIndex(-1)
    onSearch(symbol)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showDropdown) return

    if (e.key === "ArrowDown") {
      e.preventDefault()
      setSelectedIndex((prev) => (prev < suggestions.length - 1 ? prev + 1 : prev))
    } else if (e.key === "ArrowUp") {
      e.preventDefault()
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1))
    } else if (e.key === "Enter" && selectedIndex >= 0) {
      e.preventDefault()
      handleSelect(suggestions[selectedIndex].value)
    }
  }

  return (
    <div className="w-full max-w-[800px] space-y-12">
      {/* Main Headline */}
      <div className="animate-fade-in-down space-y-8 text-center">
        {/* Main Headline */}
        <h1 className="bg-gradient-to-r from-white via-green-100 to-green-300 bg-clip-text font-sans text-6xl font-bold leading-tight text-transparent">
          Track Market Sentiment in Real-Time
        </h1>

        {/* Subheadline */}
        <p className="mx-auto max-w-2xl font-sans text-xl leading-relaxed text-gray-400">
          AI-powered analysis of NSE stock news and sentiment trends
        </p>
      </div>

      <div className="animate-scale-in space-y-6">
        <div className="relative">
          <Search className="absolute left-6 top-1/2 h-6 w-6 -translate-y-1/2 text-green-500 transition-all" />
          <Input
            type="text"
            placeholder="Search NSE stocks... (e.g., RELIANCE, TCS)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            className="h-16 rounded-2xl border-2 border-gray-700/50 bg-gray-900/40 pl-16 pr-6 font-sans text-lg text-white shadow-inner backdrop-blur-lg placeholder:text-gray-500 transition-all duration-300 focus:scale-[1.02] focus:border-green-500 focus:shadow-lg focus:shadow-green-500/30 focus:ring-0"
          />
        </div>

        {showDropdown && (
          <div className="absolute top-full z-50 mt-2 max-h-[300px] w-full overflow-y-auto rounded-xl border border-gray-700/50 bg-gray-900/90 backdrop-blur-lg custom-scrollbar">
            {suggestions.map((suggestion, index) => (
              <button
                key={suggestion.symbol}
                onClick={() => handleSelect(suggestion.value)}
                className={`w-full px-4 py-3 text-left font-sans text-sm text-white transition-all ${
                  index === selectedIndex ? "bg-gray-700/50" : "hover:bg-gray-700/50"
                }`}
              >
                {suggestion.display_text}
              </button>
            ))}
          </div>
        )}

        <RecentSearches onSelect={handleSelect} />
      </div>

      <div className="animate-stagger-fade-in z-10 flex flex-wrap items-center justify-center gap-4">
        <div className="group flex items-center gap-2 rounded-full border border-gray-700/50 bg-gray-800/50 px-4 py-2 backdrop-blur-sm transition-all hover:-translate-y-1 hover:border-green-500/50 hover:shadow-lg hover:shadow-green-500/20">
          <Brain className="h-4 w-4 text-cyan-500" />
          <span className="font-sans text-sm text-gray-300 group-hover:text-white">FinBERT Powered</span>
        </div>

        <div className="group flex items-center gap-2 rounded-full border border-gray-700/50 bg-gray-800/50 px-4 py-2 backdrop-blur-sm transition-all hover:-translate-y-1 hover:border-green-500/50 hover:shadow-lg hover:shadow-green-500/20">
          <TrendingUp className="h-4 w-4 text-green-500" />
          <span className="font-sans text-sm text-gray-300 group-hover:text-white">Sentiment Analysis</span>
        </div>

        <div className="group flex items-center gap-2 rounded-full border border-gray-700/50 bg-gray-800/50 px-4 py-2 backdrop-blur-sm transition-all hover:-translate-y-1 hover:border-green-500/50 hover:shadow-lg hover:shadow-green-500/20">
          <Zap className="h-4 w-4 text-yellow-500" />
          <span className="font-sans text-sm text-gray-300 group-hover:text-white">Trading Intelligence</span>
        </div>

        <div className="group flex items-center gap-2 rounded-full border border-gray-700/50 bg-gray-800/50 px-4 py-2 backdrop-blur-sm transition-all hover:-translate-y-1 hover:border-green-500/50 hover:shadow-lg hover:shadow-green-500/20">
          <CheckCircle className="h-4 w-4 text-green-500" />
          <span className="font-sans text-sm text-gray-300 group-hover:text-white">Market Insights</span>
        </div>
      </div>

      {/* Loading/Complete State */}
      <div className="flex min-h-[140px] flex-col items-center justify-center">
        {isLoading && (
          <div className="flex flex-col items-center gap-4">
            <SegmentedRingLoader />
            <p className="animate-pulse font-mono text-sm text-yellow-400">Analyzing sentiment...</p>
          </div>
        )}

        {isComplete && !isLoading && (
          <div className="animate-pulse">
            <p className="font-sans text-lg font-medium text-green-500">✅ Analysis Complete</p>
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3">
            <p className="font-sans text-sm text-red-400">{error}</p>
          </div>
        )}
      </div>
    </div>
  )
}
