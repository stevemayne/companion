import logging
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api_models import (
    ApiErrorBody,
    ApiErrorResponse,
    ChatRequest,
    ChatResponse,
    MemoryResponse,
    SeedContextUpsertRequest,
)
from app.config import Settings, get_settings
from app.schemas import SessionSeedContext
from app.services import AppContainer, build_container


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def get_container(request: Request) -> AppContainer:
    return request.app.state.container  # type: ignore[no-any-return]


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(title=settings.app_name)
    app.state.container = build_container(settings)

    @app.exception_handler(HTTPException)
    def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        body = ApiErrorResponse(
            error=ApiErrorBody(
                code=f"http_{exc.status_code}",
                message=str(exc.detail),
            )
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        body = ApiErrorResponse(
            error=ApiErrorBody(
                code="validation_error",
                message=str(exc.errors()),
            )
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=body.model_dump(),
        )

    @app.get("/v1/health")
    def healthcheck() -> dict[str, str]:
        cfg = get_settings()
        return {"status": "ok", "environment": cfg.environment}

    @app.post(
        "/v1/chat",
        response_model=ChatResponse,
        responses={422: {"model": ApiErrorResponse}},
    )
    def chat(
        payload: ChatRequest,
        request: Request,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> ChatResponse:
        container = get_container(request)
        return container.chat_service.run_chat(request=payload, idempotency_key=idempotency_key)

    @app.get(
        "/v1/memory/{chat_session_id}",
        response_model=MemoryResponse,
        responses={422: {"model": ApiErrorResponse}},
    )
    def get_memory(chat_session_id: UUID, request: Request) -> MemoryResponse:
        container = get_container(request)
        return container.chat_service.get_memory(chat_session_id=chat_session_id)

    @app.post(
        "/v1/sessions/{chat_session_id}/seed",
        response_model=SessionSeedContext,
        status_code=status.HTTP_201_CREATED,
        responses={409: {"model": ApiErrorResponse}, 422: {"model": ApiErrorResponse}},
    )
    def create_seed_context(
        chat_session_id: UUID,
        payload: SeedContextUpsertRequest,
        request: Request,
    ) -> SessionSeedContext:
        container = get_container(request)
        return container.seed_store.create(chat_session_id=chat_session_id, payload=payload)

    @app.put(
        "/v1/sessions/{chat_session_id}/seed",
        response_model=SessionSeedContext,
        responses={404: {"model": ApiErrorResponse}, 422: {"model": ApiErrorResponse}},
    )
    def update_seed_context(
        chat_session_id: UUID,
        payload: SeedContextUpsertRequest,
        request: Request,
    ) -> SessionSeedContext:
        container = get_container(request)
        return container.seed_store.update(chat_session_id=chat_session_id, payload=payload)

    @app.get(
        "/v1/sessions/{chat_session_id}/seed",
        response_model=SessionSeedContext,
        responses={404: {"model": ApiErrorResponse}, 422: {"model": ApiErrorResponse}},
    )
    def get_seed_context(chat_session_id: UUID, request: Request) -> SessionSeedContext:
        container = get_container(request)
        seed_context = container.seed_store.get(chat_session_id=chat_session_id)
        if seed_context is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Seed context not found.",
            )
        return seed_context

    return app


app = create_app()
