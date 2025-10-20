import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const query = searchParams.get("query")

  if (!query) {
    return NextResponse.json({ error: "Query parameter is required" }, { status: 400 })
  }

  try {
    // Replace with your actual API endpoint
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/autocomplete?query=${encodeURIComponent(query)}`);

    if (!response.ok) {
      throw new Error("Failed to fetch autocomplete results")
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("[v0] Autocomplete API error:", error)
    return NextResponse.json({ error: "Failed to fetch autocomplete results" }, { status: 500 })
  }
}
