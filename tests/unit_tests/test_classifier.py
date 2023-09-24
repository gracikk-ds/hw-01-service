"""Unit tests."""

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from src.containers.containers import AppContainer


def test_predicts_not_fail(app_container: AppContainer, sample_image_np: NDArray[np.uint8]):
    """
    Test to ensure the model prediction does not fail.

    This function tries to perform prediction and prediction probability on a sample image
    to ensure that it does not throw any exception or error.

    Args:
        app_container (AppContainer): The application container holding the classification model.
        sample_image_np (NDArray[np.uint8]): The numpy array of the sample image to be tested.

    """
    planet_classifier = app_container.classfication_model()
    planet_classifier.predict(sample_image_np)


def test_predict_dont_mutate_initial_image(app_container: AppContainer, sample_image_np: NDArray[np.uint8]):
    """
    Test to ensure the initial image is not mutated during prediction.

    This function checks whether the input image is altered after making a prediction. It
    compares the initial and final states of the input image to ensure they are the same.

    Args:
        app_container (AppContainer): The application container holding the classification model.
        sample_image_np (NDArray[np.uint8]): The numpy array of the sample image to be tested.

    """
    image_to_compare = deepcopy(sample_image_np)
    planet_classifier = app_container.classfication_model()
    planet_classifier.predict(sample_image_np)

    assert np.allclose(sample_image_np, image_to_compare)  # noqa: S101
