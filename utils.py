
import torch


def fout_wrapper(output) -> torch.Tensor:
    """
    Wraps the output of a model to ensure it is a tensor.
    If the output is a tuple, it returns the first element.
    If the output is already a tensor, it returns it as is.
    """
    if isinstance(output, tuple):
        return output[0]
    return output
