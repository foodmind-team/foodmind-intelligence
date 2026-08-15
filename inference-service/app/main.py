from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from hmac import compare_digest

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from app.config import Settings
from app.model import ModelPackage, ModelPackageError
from app.schemas import (
    FailureDetail,
    InferenceFailure,
    InferenceRequest,
    InferenceSuccess,
    Prediction,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.model = None
        application.state.model_error = None
        try:
            application.state.model = ModelPackage.load(resolved.model_package_dir)
        except ModelPackageError:
            application.state.model_error = "MODEL_PACKAGE_INCOMPATIBLE"
        yield

    application = FastAPI(title="FoodMind Recommendation Inference", version="1.0.0", lifespan=lifespan)
    application.state.settings = resolved

    async def require_service_token(
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> None:
        if authorization is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing service credential")
        scheme, separator, token = authorization.partition(" ")
        expected = resolved.internal_service_token.get_secret_value()
        if scheme.lower() != "bearer" or not separator or not compare_digest(token, expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid service credential")

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready")
    async def ready(request: Request) -> dict[str, str]:
        if request.app.state.model is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="model package unavailable")
        return {"status": "ready", "modelVersion": "hybrid-ranking-v1"}

    @application.post(
        "/internal/v1/recommendations/score",
        response_model=InferenceSuccess | InferenceFailure,
        dependencies=[Depends(require_service_token)],
    )
    async def score(body: InferenceRequest, request: Request) -> InferenceSuccess | InferenceFailure:
        if body.deadline_at <= datetime.now(UTC):
            raise HTTPException(status.HTTP_408_REQUEST_TIMEOUT, detail="deadline expired")
        model: ModelPackage | None = request.app.state.model
        if model is None:
            return InferenceFailure(
                requestId=body.request_id,
                traceId=body.trace_id,
                error=FailureDetail(code="MODEL_PACKAGE_INCOMPATIBLE"),
            )
        predictions = tuple(
            Prediction(
                candidateId=candidate.candidate_id,
                probability=float(probability),
                modelScore=float(model_score),
                userCf=user_cf,
                itemCf=item_cf,
                signals=candidate.evidence,
            )
            for candidate in body.candidates
            for probability, model_score, user_cf, item_cf in (model.score(body.model_user_key, candidate),)
        )
        return InferenceSuccess(requestId=body.request_id, traceId=body.trace_id, predictions=predictions)

    return application


app = create_app()


def run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002)  # noqa: S104 - container entry point


if __name__ == "__main__":
    run()
