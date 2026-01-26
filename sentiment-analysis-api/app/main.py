from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from app.schemas import (
    BatchSentimentRequest,
    HealthResponse,
    SentimentRequest,
    SentimentResult,
)
from app.service.sentiment import SentimentService

app = FastAPI(title="Sentiment Analysis API", version="0.1.0")


@lru_cache()
def get_service() -> SentimentService:
    return SentimentService()


@app.get("/health", response_model=HealthResponse)
def health(service: SentimentService = Depends(get_service)) -> HealthResponse:
    return HealthResponse(status="ok", model=service.model_name)


@app.post("/sentiment", response_model=SentimentResult)
def analyze_sentiment(
    payload: SentimentRequest, service: SentimentService = Depends(get_service)
) -> SentimentResult:
    try:
        result = service.analyze_text(payload.text, language=payload.language)
        return SentimentResult(**result)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/sentiment/batch", response_model=list[SentimentResult])
def analyze_batch(
    payload: BatchSentimentRequest, service: SentimentService = Depends(get_service)
) -> list[SentimentResult]:
    try:
        results = service.analyze_batch(payload.texts, language=payload.language)
        return [SentimentResult(**item) for item in results]
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
