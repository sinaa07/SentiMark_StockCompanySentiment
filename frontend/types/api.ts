export interface AutocompleteResult {
  symbol: string
  company_name: string
  display_text: string
  series: string
  value: string
}

export interface RecentSearch {
  symbol: string
  company_name: string
  timestamp: string
  series: string
}

export interface Company {
  symbol: string
  name: string
  sector: string
}

export interface SentimentSummary {
  bullish: number
  neutral: number
  bearish: number
  overall_score: number
  sentiment_label: string
  confidence: number
}

export interface Article {
  title: string
  url: string
  source: string
  published: string
  sentiment: string
  positive: number
  neutral: number
  negative: number
}

export interface PipelineResponse {
  company: Company
  article_count: number
  sentiment_summary: SentimentSummary
  articles: Article[]
}
