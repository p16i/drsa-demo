import torch
import numpy as np

from cxai import utils as putils

device = putils.get_device()
torch.manual_seed(1)


def get_test_image_path(filename: str) -> str:
    return f"./tests/data/{filename}"


def _generate_input(filename: str, input_size: tuple, input_transform) -> torch.Tensor:

    if "noise" in filename:
        if "big" in filename:
            input_size = [input_size[0]] + (np.array(input_size)[1:] * 2).tolist()
            return torch.randn(input_size).to(device)
        else:
            return torch.randn(input_size).to(device)
    else:
        imgpath = get_test_image_path(filename)
        img = putils.image.load_image(imgpath)

        return input_transform(img).to(device)


def _find_biases_and_set_to_zero(module: torch.nn.Module, verbose=False):

    for child in module.children():
        if hasattr(child, "bias"):
            # only leave nodes can have bias?
            if verbose:
                print(f"set bias to zero [{child}]")
            child.bias = torch.nn.Parameter(torch.zeros_like(child.bias))
        else:
            _find_biases_and_set_to_zero(child)
