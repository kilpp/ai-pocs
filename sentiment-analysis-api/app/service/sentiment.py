from typing import List, Optional

from transformers import AutoTokenizer, pipeline

from app.config import get_settings


class SentimentService:
    """Wraps a transformers sentiment pipeline with sane defaults."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.model_name
        self.language_hint: Optional[str] = None
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=settings.model_revision
        )
        self.pipeline = pipeline(
            task=settings.pipeline_task,
            model=self.model_name,
            tokenizer=tokenizer,
            revision=settings.model_revision,
            device=settings.device,
            truncation=True,
        )
        self.max_length = settings.max_length
        self.batch_size = settings.batch_size

    def _normalize(self, label: str) -> str:
        # Common multilingual models return either sentiment words or star ratings.
        cleaned = label.lower()
        if cleaned.startswith("star") or cleaned.startswith("label_"):
            # nlptown/bert-base-multilingual-uncased-sentiment returns 1-5 stars.
            try:
                stars = int("".join(ch for ch in cleaned if ch.isdigit()))
                if stars <= 2:
                    return "negative"
                if stars == 3:
                    return "neutral"
                return "positive"
            except ValueError:
                return cleaned
        return cleaned

    def analyze_text(self, text: str, language: Optional[str] = None) -> dict:
        result = self.pipeline(
            text,
            truncation=True,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )[0]
        return {
            "label": self._normalize(result["label"]),
            "score": float(result["score"]),
            "model": self.model_name,
            "language": language,
        }

    def analyze_batch(
        self, texts: List[str], language: Optional[str] = None
    ) -> List[dict]:
        outputs = self.pipeline(
            texts,
            truncation=True,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        return [
            {
                "label": self._normalize(item["label"]),
                "score": float(item["score"]),
                "model": self.model_name,
                "language": language,
            }
            for item in outputs
        ]
