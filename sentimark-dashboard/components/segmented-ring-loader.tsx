export default function SegmentedRingLoader() {
  return (
    <div className="relative h-20 w-20">
      <svg className="h-full w-full animate-spin" viewBox="0 0 100 100" style={{ animationDuration: "2s" }}>
        <defs>
          <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="100%" stopColor="#fbbf24" />
          </linearGradient>
        </defs>
        <circle
          cx="50"
          cy="50"
          r="40"
          fill="none"
          stroke="url(#gradient)"
          strokeWidth="8"
          strokeDasharray="10 5"
          strokeLinecap="round"
        />
      </svg>
    </div>
  )
}
