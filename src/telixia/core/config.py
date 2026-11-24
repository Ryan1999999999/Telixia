"""Central configuration, read from .env (never hard-code secrets)."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    postgres_db: str = "telixia"
    postgres_user: str = "telixia_admin"
    postgres_password: SecretStr
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    telixia_ro_password: SecretStr

    qdrant_host: str = "localhost"
    qdrant_http_port: int = 6333
    qdrant_api_key: SecretStr

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    def pg_params(self, read_only: bool = False) -> dict:
        """Connection parameters for pg8000 (pure-Python PostgreSQL driver)."""
        return {
            "user": "telixia_ro" if read_only else self.postgres_user,
            "password": (self.telixia_ro_password if read_only else self.postgres_password).get_secret_value(),
            "host": self.postgres_host,
            "port": self.postgres_port,
            "database": self.postgres_db,
            "timeout": 5,
        }

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_http_port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()