import logging
import re
from asyncio import sleep
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.api_models import (
    ApiErrorBody,
    ApiErrorResponse,
    ChatRequest,
    ChatResponse,
    MemoryResponse,
    SeedContextUpsertRequest,
)
from app.config import Settings, get_settings
from app.observability import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    InMemoryRateLimiter,
    metrics_payload,
)
from app.safety import contains_prompt_injection, redact_pii
from app.schemas import SessionSeedContext
from app.services import AppContainer, build_container


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def get_container(request: Request) -> AppContainer:
    return request.app.state.container  # type: ignore[no-any-return]


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            app.state.container.agent_dispatcher.shutdown()

    app = FastAPI(title=app_settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip() for origin in app_settings.cors_allow_origins.split(",") if origin
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.container = build_container(app_settings)
    app.state.rate_limiter = InMemoryRateLimiter(
        limit_per_minute=app_settings.rate_limit_per_minute
    )

    @app.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id

        path = request.url.path
        protected = path.startswith("/v1") and path != "/v1/health"

        if protected and app_settings.enable_api_key_auth:
            api_key = request.headers.get("X-API-Key")
            if api_key != app_settings.service_api_key:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content=ApiErrorResponse(
                        error=ApiErrorBody(code="http_401", message="Unauthorized")
                    ).model_dump(),
                    headers={"X-Request-ID": request_id},
                )

        if protected and app_settings.enable_rate_limit:
            fallback_key = request.client.host if request.client else "unknown"
            key = request.headers.get("X-API-Key", fallback_key)
            if not app.state.rate_limiter.allow(key):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content=ApiErrorResponse(
                        error=ApiErrorBody(code="http_429", message="Rate limit exceeded")
                    ).model_dump(),
                    headers={"X-Request-ID": request_id},
                )

        start = perf_counter()
        response = await call_next(request)
        duration = perf_counter() - start

        response.headers["X-Request-ID"] = request_id
        REQUEST_COUNT.labels(
            method=request.method,
            path=path,
            status=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, path=path).observe(duration)

        logging.getLogger("aether.http").info(
            "request.complete method=%s path=%s status=%s duration_ms=%.2f request_id=%s",
            request.method,
            path,
            response.status_code,
            duration * 1000,
            request_id,
        )
        return response

    @app.exception_handler(HTTPException)
    def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        body = ApiErrorResponse(
            error=ApiErrorBody(
                code=f"http_{exc.status_code}",
                message=str(exc.detail),
            )
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(),
            headers={"X-Request-ID": getattr(request.state, "request_id", "unknown")},
        )

    @app.exception_handler(RequestValidationError)
    def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        body = ApiErrorResponse(
            error=ApiErrorBody(
                code="validation_error",
                message=str(exc.errors()),
            )
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=body.model_dump(),
            headers={"X-Request-ID": getattr(request.state, "request_id", "unknown")},
        )

    @app.get("/v1/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok", "environment": app_settings.environment}

    @app.get("/metrics")
    def metrics() -> Response:
        body, content_type = metrics_payload()
        return Response(content=body, media_type=content_type)

    def sse_event(event: str, payload: dict[str, str | int | bool | None]) -> str:
        import json

        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    @app.post(
        "/v1/chat",
        response_model=ChatResponse,
        responses={400: {"model": ApiErrorResponse}, 422: {"model": ApiErrorResponse}},
    )
    def chat(
        payload: ChatRequest,
        request: Request,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> ChatResponse:
        if contains_prompt_injection(payload.message):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Potential prompt injection detected",
            )

        sanitized = ChatRequest(
            chat_session_id=payload.chat_session_id,
            message=redact_pii(payload.message),
        )

        container = get_container(request)
        return container.chat_service.run_chat(request=sanitized, idempotency_key=idempotency_key)

    @app.get(
        "/v1/chat/stream",
        responses={400: {"model": ApiErrorResponse}, 422: {"model": ApiErrorResponse}},
    )
    async def chat_stream(
        chat_session_id: UUID,
        message: str,
        request: Request,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> StreamingResponse:
        if contains_prompt_injection(message):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Potential prompt injection detected",
            )

        sanitized = ChatRequest(
            chat_session_id=chat_session_id,
            message=redact_pii(message),
        )
        container = get_container(request)
        chat_result = container.chat_service.run_chat(
            request=sanitized,
            idempotency_key=idempotency_key,
        )
        request_id = getattr(request.state, "request_id", str(uuid4()))

        async def event_generator() -> AsyncIterator[str]:
            yield sse_event(
                "start",
                {
                    "chat_session_id": str(chat_result.chat_session_id),
                    "message_id": str(chat_result.assistant_message.message_id),
                    "request_id": request_id,
                    "seed_version": chat_result.seed_version,
                    "idempotency_replay": chat_result.idempotency_replay,
                },
            )
            chunks = re.findall(r"\S+\s*", chat_result.assistant_message.content)
            for chunk in chunks:
                yield sse_event("delta", {"chunk": chunk})
                await sleep(0.01)
            yield sse_event(
                "done",
                {
                    "chat_session_id": str(chat_result.chat_session_id),
                    "message_id": str(chat_result.assistant_message.message_id),
                },
            )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

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
