import torch

from .linear_quantize import linear_quantize


def linear_quantize_bias_per_output_channel(bias, weight_scale, input_scale):
    """
    linear quantization for single bias tensor
        quantized_bias = fp_bias / bias_scale

    Args:
        bias: [torch.FloatTensor] bias weight to be quantized
        weight_scale: [float or torch.FloatTensor] weight scale tensor
        input_scale: [float] input scale

    Returns:
        [torch.IntTensor] quantized bias tensor
    """
    assert bias.dim() == 1
    assert (bias.dtype == torch.float)
    assert (isinstance(input_scale, float))
    if isinstance(weight_scale, torch.Tensor):
        assert (weight_scale.dtype == torch.float)
        weight_scale = weight_scale.view(-1)
        assert (bias.numel() == weight_scale.numel())

    bias_scale = weight_scale * input_scale
    qbias = linear_quantize(bias, 32, bias_scale, 0, dtype=torch.int32)
    return qbias, bias_scale, 0
