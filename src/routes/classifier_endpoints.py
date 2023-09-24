"""
This module provides the prediction endpoint for a classification service.

It sets up the logging configurations and defines the `predict` endpoint. The `predict` endpoint
takes an image as input and returns the predicted classes by using a classification model.
The classification model is provided by the `TorchWrapper` service which is injected as a dependency.
"""

import sys
from logging.handlers import SysLogHandler

import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File
from loguru import logger

from src.containers.containers import AppContainer
from src.logger.log import DevelopFormatter
from src.routes.routers import classifier_router
from src.service.classifier import TorchWrapper

logger.remove()
develop_fmt = DevelopFormatter("InferenceService")
logger.add(sys.stderr, format=develop_fmt)  # type: ignore
syslog_handler = SysLogHandler(address=("127.0.0.1", 9000))
logger.add(syslog_handler, format=develop_fmt)  # type: ignore


@classifier_router.post("/predict")
@inject
def predict(
    image: bytes = File(),
    service: TorchWrapper = Depends(Provide[AppContainer.classfication_model]),
):
    """
    Make a prediction on the given image using the provided classification model.

    This endpoint takes an image file in bytes and uses the `TorchWrapper` service
    to make a prediction and return the predicted classes.

    Args:
        image (bytes): The image file in bytes to make predictions on.
        service (TorchWrapper): The classification service to use for making predictions.

    Returns:
        dict: A dictionary with the key 'classes' containing the sorted classes by confidence.
    """
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    class_indexes = service.predict(img)[0]
    class_indexes = [int(_) for _ in reversed(class_indexes.argsort())]
    return {"classes": class_indexes}
