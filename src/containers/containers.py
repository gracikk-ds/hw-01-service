# pylint: disable=c-extension-no-member,no-name-in-module

"""Containers for injection."""

from dependency_injector import containers
from dependency_injector.providers import Configuration, Singleton

from src.service.classifier import TorchWrapper


class AppContainer(containers.DeclarativeContainer):
    """
    Dependency injection container for managing application components.

    Args:
        containers.DeclarativeContainer: The base class for the dependency injection container.
    """

    config = Configuration()

    """
    Singleton provider for the Classification Model component.

    Returns:
        checkpoint (str): path to model weights.
        device (str): device type
    """
    classfication_model: Singleton[TorchWrapper] = Singleton(
        TorchWrapper,
        checkpoint=config.classification_model.checkpoint,
        device=config.classification_model.device,
    )
