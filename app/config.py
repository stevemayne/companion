from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Project Aether API"
    environment: str = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    use_external_stores: bool = False
    postgres_dsn: str = "postgresql://aether:aether@localhost:5432/aether"
    qdrant_url: str = "http://localhost:6333"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    inference_provider: str = "mock"
    inference_model: str = "local-model"
    inference_base_url: str = "http://localhost:1234/v1"
    inference_api_key: str | None = None
    inference_timeout_seconds: float = 30.0
    inference_max_retries: int = 2
    inference_temperature: float = 0.75
    inference_frequency_penalty: float = 0.4
    inference_presence_penalty: float = 0.3
    inference_max_tokens: int = 512
    inference_failover_enabled: bool = False
    fallback_inference_model: str | None = None
    fallback_inference_base_url: str | None = None
    fallback_inference_api_key: str | None = None

    analysis_provider: str = "heuristic"
    analysis_model: str | None = None
    analysis_base_url: str | None = None
    analysis_api_key: str | None = None
    analysis_timeout_seconds: float = 8.0
    analysis_max_retries: int = 1

    enable_background_agents: bool = True
    enable_api_key_auth: bool = False
    service_api_key: str = "change-me"
    enable_rate_limit: bool = True
    rate_limit_per_minute: int = 120
    cors_allow_origins: str = "http://localhost:5173"
    debug_tracing: bool = False
    debug_trace_limit: int = 100

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
