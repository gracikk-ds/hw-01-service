# pylint: disable=wildcard-import,unused-wildcard-import,unused-import
"""App Entrypoint."""

from fastapi import FastAPI
from omegaconf import OmegaConf
from prometheus_client import start_http_server

from src.containers.containers import AppContainer
from src.routes import classifier_endpoints, health_endpoints
from src.routes.routers import classifier_router, health_router
from src.settings import app_settings


def create_app() -> FastAPI:
    """
    Create a FastAPI instance with configured routes.

    Returns:
        FastAPI: An instance of the FastAPI application.
    """
    start_http_server(app_settings.prometheus_port)

    container = AppContainer()
    cfg = OmegaConf.load("configs/config.yml")
    container.config.from_dict(cfg)  # type: ignore
    container.wire([classifier_endpoints])

    app: FastAPI = FastAPI(
        title=app_settings.component_name,
        version=app_settings.service_version,
        description="Inference service for classification model.",
    )

    app.include_router(health_router, prefix="/health", tags=["health"])
    app.include_router(classifier_router, prefix="/classifier", tags=["classifier"])
    return app


app = create_app()
