import { type NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  try {
    // Replace with your actual API endpoint
    const response = await fetch("http://localhost:8000/api/recent", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
      cache: "no-store", // avoid caching for always-fresh recent searches
    });

    if (!response.ok) {
      throw new Error("Failed to fetch recent searches");
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("[v0] Recent API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch recent searches" },
      { status: 500 }
    );
  }
}