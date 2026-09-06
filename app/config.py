"""Settings, read from the environment or a .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fixed by the embedding model. No point making these configurable.
EMBEDDING_SOURCE_REPO = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_FILE = "onnx/model.onnx"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 512
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CHUNK_TARGET_TOKENS = 400
CHUNK_MIN_TOKENS = 24

RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 8
RERANK_MAX_CHARS = 600
RRF_K = 60
RECITAL_WEIGHT = 0.5

# Cross-encoder logits. Anything the corpus covers scores well above -8, and an
# off-topic question sits near -11 on every candidate.
RELEVANCE_THRESHOLD = 0.0
OUT_OF_SCOPE_SCORE = -8.0
MAX_CHUNKS_PER_PROVISION = 2

SUBTASK_WORKERS = 3
MAX_SUBTASKS = 4


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"

    llm_provider: Literal["dummy", "ollama"] = "dummy"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:0.5b-instruct"
    ollama_timeout_s: float = 180.0

    embedding_model: str = "BAAI/bge-small-en-v1.5-fp32"

    dense_k: int = 20
    sparse_k: int = 20
    fused_k: int = 20
    rerank_top_n: int = 3
    prefer_current_law: bool = True
    max_context_tokens: int = 1200

    max_retries: int = 1
    recursion_limit: int = Field(default=15, gt=0)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def tokenizer_dir(self) -> Path:
        return self.data_dir / "tokenizer" / "bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
