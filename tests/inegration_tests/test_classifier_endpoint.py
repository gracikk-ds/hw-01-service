"""This module contains tests for predicting genres using a FastAPI application.

The tests use FastAPI's TestClient to send HTTP requests to the application,
and then check the responses to ensure that they are correct.
"""

from http import HTTPStatus

from fastapi.testclient import TestClient


def test_predict(client: TestClient, sample_image_bytes: bytes):
    """Test the predict endpoint of the classifier in the FastAPI application.

    This test sends a POST request to the "/classifier/predict" endpoint of the
    FastAPI application, including a sample image in the request data.
    It then checks that the response has a 200 status code, and that the response
    data includes a list of predicted classes.

    Args:
        client (TestClient): The test client used to send requests to the application.
        sample_image_bytes (bytes): The byte representation of a sample image.

    Raises:
        AssertionError: If the response status code is not 200, or if the response
            data does not include a list of predicted classes.
    """
    files = {
        "image": sample_image_bytes,
    }

    response = client.post("/classifier/predict", files=files)
    assert response.status_code == HTTPStatus.OK  # noqa: S101

    predicted_classes = response.json()["classes"]
    assert isinstance(predicted_classes, list)  # noqa: S101
