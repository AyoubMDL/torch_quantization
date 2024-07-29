import torch

from .linear_quantize import get_quantized_range, linear_quantize


def get_quantization_scale_for_weight(weight, bitwidth):
    r_max = weight.abs().max().item()

    _, quantized_max = get_quantized_range(bitwidth)

    return r_max / quantized_max


def linear_quantize_weight_per_channel(tensor, bitwidth):
    """
    Recall that for 2D convolution, the weight tensor is a 4-D tensor
    in the shape of (num_output_channels, num_input_channels, kernel_height, kernel_width)
    """
    num_out_channel = tensor.shape[0]
    scale = torch.zeros(num_out_channel, device="cpu")

    for oc in range(num_out_channel):
        tensor_slice = tensor[oc, ...]
        scale[oc] = get_quantization_scale_for_weight(tensor_slice, bitwidth)

    scale_shape = [1] * tensor.dim()
    scale_shape[0] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_quantize(tensor, bitwidth, scale, zero_point=0)

    return quantized_tensor, scale, 0
