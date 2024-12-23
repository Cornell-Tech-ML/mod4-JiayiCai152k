from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")
    new_height = height // kh
    new_width = width // kw

    # Reshape the tensor to group pooling windows
    tiled = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    tiled = tiled.permute(0, 1, 2, 4, 3, 5).contiguous()
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    tiled, new_height, new_width = tile(input, kernel)

    # Calculate mean along the last dimension (pooling window size)
    pooled = tiled.sum(dim=4) / tiled.shape[4]

    # Reshape to final output dimensions
    return pooled.view(tiled.shape[0], tiled.shape[1], new_height, new_width)


# 4.4

max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax"""
    max_values = max_reduce(input, dim)
    return input == max_values


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max"""
        dim_int = int(dim.item())
        max_values = max_reduce(input, dim_int)
        ctx.save_for_backward(input, max_values, dim_int)
        return max_values

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max"""
        input, max_values, dim_int = ctx.saved_values
        mask = input == max_values
        return grad_output * mask, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max values along the specified dimension."""
    return Max.apply(input, tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Computes the softmax for the input tensor along the specified dimension."""
    exp_input = input.exp()
    sum_exp = exp_input.sum(dim=dim)
    return exp_input / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Computes the log of the softmax for the input tensor along the specified dimension."""
    max_input = max_reduce(input, dim)
    shifted = input - max_input
    log_sum_exp = shifted.exp().sum(dim).log()
    return shifted - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Performs max pooling on the input tensor using the specified kernel size."""
    # batch, channel, height, width = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    max_values = max_reduce(tiled, 4)
    return max_values.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Applies dropout to the input tensor with a given drop rate and scaling."""
    if ignore:
        return input

    # Handle edge case where rate == 1.0
    if rate >= 1.0:
        return input.zeros(input.shape)

    mask = rand(input.shape) > rate
    scale = 1.0 / (1.0 - rate)
    return input * mask * scale
