"""Module provides functionality to convert PyTorch model to other formats."""

import os
from typing import List

import torch

INPUT_RESOLUTION: int = 224


class Converter:
    """A class to handle conversion from PyTorch models to other formats.

    Attributes:
        device (str): The device to run the model on.
        checkpoint (str): The path to the PyTorch model checkpoint.
        model (torch.nn.Module): The loaded PyTorch model.
    """

    def __init__(self, checkpoint: str, device: str = "cpu"):
        """
        Initialize the Converter class by loading the model and moving it to the specified device.

        Args:
            checkpoint (str): The path to the PyTorch model checkpoint.
            device (str): The device to run the model on. Defaults to "cpu".

        Attributes:
            device (str): The device to run the model on.
            checkpoint (str): The path to the PyTorch model checkpoint.
            model (torch.nn.Module): The loaded PyTorch model.
        """
        self.device = device
        self.checkpoint = checkpoint

        self.model = torch.jit.load(checkpoint, map_location=device)  # type: ignore
        self.model.eval()
        self.model.to(self.device)

    def torch_to_onnx(self, dummy_input: torch.Tensor, input_names: List[str], output_names: List[str]):
        """
        Convert the PyTorch model to ONNX format.

        This method exports the loaded PyTorch model to ONNX format by using the specified dummy input, input names,
        and output names. The exported ONNX model is saved in the same directory as the original PyTorch checkpoint
        with the name "model.onnx".

        Args:
            dummy_input (torch.Tensor): A tensor of the right shape and type to perform a forward pass on the model.
            input_names (List[str]): List of names of the input tensors for the model.
            output_names (List[str]): List of names of the output tensors for the model.
        """
        torch.onnx.export(
            self.model,
            dummy_input,
            os.path.join(os.path.dirname(self.checkpoint), "model.onnx"),
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=9,
        )


if __name__ == "__main__":
    traced_input = torch.rand(1, 3, INPUT_RESOLUTION, INPUT_RESOLUTION)
    converter = Converter(checkpoint="weights/classifier.pt")
    converter.torch_to_onnx(traced_input, ["input"], ["output"])
