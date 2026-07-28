# FastAPI application entry point — owns app creation and route registration.
# The app-factory pattern keeps the ASGI instance lazily constructed and testable.
from fastapi import FastAPI


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance.

    Uses the app-factory pattern so tests and alternative entry points can
    construct fresh, isolated instances without relying on module-level state.

    Returns:
        FastAPI: A fully configured but not-yet-running ASGI application.

    Raises:
        None — this function performs no I/O and always succeeds.
    """
    # Create the core FastAPI instance with metadata visible in /docs.
    application = FastAPI(
        title="FoodMind Cooking Plan Agent",
        version="1.0.0",
    )

    # Health endpoint: used by orchestrators (e.g. Spring Boot) to verify
    # service liveness before forwarding traffic.
    @application.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "alive"}

    return application


# Module-level singleton: the ASGI callable that uvicorn (or any server) imports.
app = create_app()
