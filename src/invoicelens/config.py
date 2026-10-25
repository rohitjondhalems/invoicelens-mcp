"""
All settings come from environment variables (prefix INVOICELENS_) or a .env file.

Providers are chosen independently for embeddings and extraction:
  - "azure"  : Azure OpenAI (default for a live demo)
  - "openai" : OpenAI API
  - "local"  : offline, no keys — hashing embedder / rule-based extractor.
               Used by tests and CI, and as a fallback so the server always starts.
  - "auto"   : azure if Azure credentials are set, else openai if an OpenAI key
               is set, else local.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["auto", "azure", "openai", "local"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INVOICELENS_", env_file=".env", extra="ignore")

    data_dir: Path = Field(Path("./data"), description="Where FAISS indexes, metadata and PDFs are stored")
    default_collection: str = "default"

    # Providers
    embedding_provider: Provider = "auto"
    extraction_provider: Provider = "auto"

    # Azure OpenAI
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_embedding_deployment: str = "text-embedding-3-large"
    azure_chat_deployment: str = "gpt-4o"

    # OpenAI
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-large"
    openai_chat_model: str = "gpt-4o"

    # Hybrid chunking
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # Safety limits
    max_pdf_bytes: int = 20 * 1024 * 1024
    max_pages: int = 200
    allow_local_paths: bool = Field(
        True,
        description="Let tools read PDFs from a server-side path. Turn OFF when the "
        "server is reachable over a network: clients should send base64 instead.",
    )

    # Optional bearer token for the HTTP transports (MCP over HTTP and REST)
    api_token: str | None = None

    # External data tools (optional — the tools report clearly when a key is missing)
    tavily_api_key: str | None = Field(None, description="Web search: https://app.tavily.com")
    fmp_api_key: str | None = Field(None, description="Market data: https://site.financialmodelingprep.com")
    external_timeout_s: float = 15.0
    market_data_cache_s: int = Field(600, description="Cache FMP responses (free plan: 250 calls/day)")

    def resolve(self, provider: Provider) -> Literal["azure", "openai", "local"]:
        if provider != "auto":
            return provider
        if self.azure_openai_endpoint and self.azure_openai_api_key:
            return "azure"
        if self.openai_api_key:
            return "openai"
        return "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()
