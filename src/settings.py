"""This module defines the AppSettings class which encapsulates settings related to App configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Encapsulates settings related to App configuration."""

    # setup model config
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # app related settings
    prometheus_port: int = Field(description="Port for Prometheus metrics")
    service_version: str = Field("0.0.1", description="Service version")
    component_name: str = Field("InferenceService", description="Service name")

    # inference related settings
    base_config_path: str = Field("configs/config.yml", description="Base configuration file")


# Create an instance of the AppSettings class
app_settings = AppSettings()  # type: ignore
