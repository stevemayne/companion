import logging

from fastapi import FastAPI

from app.config import Settings, get_settings


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(title=settings.app_name)

    @app.get("/v1/health")
    def healthcheck() -> dict[str, str]:
        cfg = get_settings()
        return {"status": "ok", "environment": cfg.environment}

    return app


app = create_app()
