"use client"

import { useEffect, useState } from "react"
import { Clock } from "lucide-react"
import type { RecentSearch } from "@/types/api"

interface RecentSearchesProps {
  onSelect: (symbol: string) => void
}

export default function RecentSearches({ onSelect }: RecentSearchesProps) {
  const [searches, setSearches] = useState<RecentSearch[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchRecentSearches = async () => {
      try {
        const response = await fetch("/api/recent")
        if (response.ok) {
          const data = await response.json()
          setSearches(data)
        }
      } catch (error) {
        console.error("[v0] Failed to fetch recent searches:", error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchRecentSearches()
  }, [])

  if (isLoading || searches.length === 0) {
    return null
  }

  return (
    <div className="w-full space-y-3">
      <div className="flex items-center gap-2">
        <Clock className="h-4 w-4 text-gray-400" />
        <h3 className="font-sans text-sm text-gray-400">Recent Searches</h3>
      </div>

      <div className="flex flex-wrap gap-3 overflow-x-auto pb-2">
        {searches.map((search) => (
          <button
            key={`${search.symbol}-${search.timestamp}`}
            onClick={() => onSelect(search.symbol)}
            className="group flex items-center gap-2 whitespace-nowrap rounded-full border border-gray-700 bg-gray-800 px-4 py-2 transition-all hover:-translate-y-0.5 hover:border-green-500/50 hover:bg-gray-700 hover:shadow-lg hover:shadow-green-500/20"
          >
            <span className="font-mono text-sm font-bold text-green-500 group-hover:text-green-400">
              {search.symbol}
            </span>
            <span className="font-sans text-xs text-gray-400 group-hover:text-gray-300">-</span>
            <span className="max-w-[200px] truncate font-sans text-xs text-gray-400 group-hover:text-gray-300">
              {search.company_name}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
