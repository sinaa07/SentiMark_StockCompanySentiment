from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime
from app.core.user_input_processor import UserInputProcessor
from app.core.pipeline import SentimentPipeline

router = APIRouter()

# ✅ Create shared instances (persist for app lifetime)
processor = UserInputProcessor()
pipeline = SentimentPipeline()

class PipelineRequest(BaseModel):
    symbol: str


@router.get("/autocomplete")
def autocomplete(query: str = Query(..., min_length=1)):
    """Get autocomplete suggestions for company names/symbols"""
    print(f"[Autocomplete] Query: {query}")
    
    if not query or len(query.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": True,
                "error_type": "invalid_query",
                "error_message": "Query parameter cannot be empty",
                "details": {"query": query},
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
    
    suggestions = processor.get_autocomplete_suggestions(query)
    return suggestions or []


@router.post("/pipeline")
def run_pipeline(request: PipelineRequest):
    """Execute sentiment analysis pipeline for a given symbol"""
    print(f"[Pipeline] Symbol: {request.symbol}")
    
    try:
        result = pipeline.run(request.symbol)

        # 📝 Cache the search in shared processor for /recent
        processor.recent_searches.append({
            "symbol": request.symbol,
            "company_name": result["company"]["name"],
            "series": "EQ",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

        return result
        
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": True,
                "error_type": "ticker_not_found",
                "error_message": "Company not found in database",
                "details": {"symbol": request.symbol},
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "error_type": "pipeline_error",
                "error_message": "Pipeline execution failed",
                "details": {"symbol": request.symbol, "error": str(e)},
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@router.get("/recent")
def get_recent_searches():
    """Retrieve recent user searches"""
    print("[Recent] Fetching recent searches")
    recent = processor.get_recent_searches()
    return recent or []
