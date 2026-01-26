from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    model_name: str = Field(
        default="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        description="Hugging Face model id or local path",
    )
    model_revision: Optional[str] = Field(
        default=None, description="Optional model revision to pin"
    )
    device: int = Field(
        default=-1,
        description="Device index for inference; -1 for CPU, 0+ for GPU",
    )
    pipeline_task: str = Field(
        default="text-classification", description="Transformers pipeline task"
    )
    max_length: int = Field(
        default=512, description="Maximum sequence length for truncation"
    )
    batch_size: int = Field(default=8, description="Batch size for inference")

    class Config:
        env_prefix = "SA_"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
