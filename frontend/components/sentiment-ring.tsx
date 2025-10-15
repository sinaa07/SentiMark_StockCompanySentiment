"use client"

import { useEffect, useState } from "react"

interface SentimentRingProps {
  value: number // 0-100
  label: string
  title: string
  color: "green" | "red" | "yellow"
  isPercentage?: boolean
}

export default function SentimentRing({ value, label, title, color, isPercentage = false }: SentimentRingProps) {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    // Animate progress
    const timer = setTimeout(() => setProgress(value), 100)
    return () => clearTimeout(timer)
  }, [value])

  const colorMap = {
    green: "#10b981",
    red: "#ef4444",
    yellow: "#fbbf24",
  }

  const strokeColor = colorMap[color]
  const size = 140
  const strokeWidth = 12
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (progress / 100) * circumference

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="rotate-[-90deg]">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="#374151"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            strokeDasharray={`${circumference} ${circumference}`}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{
              transition: "stroke-dashoffset 1s ease",
            }}
          />
        </svg>

        {/* Center value */}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="font-mono text-2xl font-bold" style={{ color: strokeColor }}>
            {isPercentage ? `${value.toFixed(1)}%` : label}
          </span>
        </div>
      </div>

      {/* Label */}
      <p className="mt-4 text-center font-sans text-sm text-gray-400">{title}</p>
      {!isPercentage && (
        <p className="mt-1 text-center font-sans text-xs font-semibold capitalize" style={{ color: strokeColor }}>
          {label}
        </p>
      )}
    </div>
  )
}
