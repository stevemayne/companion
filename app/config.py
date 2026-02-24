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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
