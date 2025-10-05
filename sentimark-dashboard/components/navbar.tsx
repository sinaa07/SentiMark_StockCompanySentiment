export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 mb-6 border-b border-gray-800 bg-black/80 backdrop-blur-lg">
      <div className="mx-auto flex h-20 max-w-7xl items-center justify-center gap-4 px-4">
        <h1 className="bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text font-sans text-4xl font-bold text-transparent drop-shadow-[0_0_20px_rgba(16,185,129,0.5)] transition-all hover:drop-shadow-[0_0_30px_rgba(16,185,129,0.7)]">
          SentiMark
        </h1>
        <div className="flex flex-col items-start gap-1">
          <p className="font-sans text-xs uppercase tracking-wider text-gray-400 transition-colors hover:text-green-400">
            NSE Market News Sentiment Dashboard
          </p>
          <div className="h-0.5 w-full bg-green-500" />
        </div>
      </div>
    </nav>
  )
}
