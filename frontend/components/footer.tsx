export default function Footer() {
  return (
    <footer className="border-t border-gray-800 bg-black">
      <div className="mx-auto max-w-7xl px-4 py-12">
        <div className="flex flex-col items-center gap-4 text-center">
          {/* Copyright */}
          <p className="font-sans text-sm text-gray-400">© 2025 SentiMark. All rights reserved.</p>

          {/* Creator Credit */}
          <p className="font-sans text-sm text-gray-500">Built with AI-powered sentiment analysis</p>

          {/* Disclaimer */}
          <p className="max-w-2xl font-sans text-xs text-gray-600">
            For informational purposes only. Not financial advice. Always conduct your own research before making
            investment decisions.
          </p>

          {/* Optional Green Accent Line */}
          <div className="mt-4 h-1 w-24 rounded-full bg-gradient-to-r from-green-500 to-cyan-500" />
        </div>
      </div>
    </footer>
  )
}
