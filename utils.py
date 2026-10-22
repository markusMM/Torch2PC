
import os
import torch


NN_BLOCK_OUT_INDEX = int(os.environ.get('NN_BLOCK_OUT_INDEX', '0'))  # Default to 0 if not set

def fout_wrapper(output: tuple[torch.Tensor]|torch.Tensor) -> torch.Tensor:
    """
    Wraps the output of a model to ensure it is a tensor.
    If the output is a tuple, it returns the first element.
    If the output is already a tensor, it returns it as is.
    """
    if isinstance(output, tuple):
        return output[NN_BLOCK_OUT_INDEX]
    return output
