import torch

from ..quantization import float_to_fixed_point, get_quantized_range


def shift_quantized_linear_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    For quantized fully-connected layer, we first precompute  Qbias.
    Recall that  Qbias = qbias - Linear[Zinput, qweight] .

    shift quantized bias to incorporate input_zero_point for nn.Linear
        shifted_quantized_bias = quantized_bias - Linear(input_zero_point, quantized_weight)

    Args:
        quantized_bias: [torch.IntTensor] quantized bias (torch.int32)
        quantized_weight: [torch.CharTensor] quantized weight (torch.int8)
        input_zero_point: [int] input zero point

    Returns:
        [torch.IntTensor] shifted quantized bias tensor
    """
    assert (quantized_bias.dtype == torch.int32)
    assert (isinstance(input_zero_point, int))

    return quantized_bias - quantized_weight.sum(1).to(torch.int32) * input_zero_point


def quantized_linear(qinput, qweight, qbias, feature_bitwidth, weight_bitwidth,
                     input_zero_point, output_zero_point,
                     input_scale, weight_scale, output_scale):
    """
    qoutput = (Linear[qinput, qweight] + Qbias) ⋅ (SinputSweight/Soutput) + Zoutput
    quantized fully-connected layer

    Args:
        input: [torch.CharTensor] quantized input (torch.int8)
        weight: [torch.CharTensor] quantized weight (torch.int8)
        bias: [torch.IntTensor] shifted quantized bias or None (torch.int32)
        feature_bitwidth: [int] quantization bit width of input and output
        weight_bitwidth: [int] quantization bit width of weight
        input_zero_point: [int] input zero point
        output_zero_point: [int] output zero point
        input_scale: [float] input feature scale
        weight_scale: [torch.FloatTensor] weight per-channel scale
        output_scale: [float] output feature scale

    Returns:
        [torch.CharIntTensor] quantized output feature (torch.int8)
    """
    assert (qinput.dtype == torch.int8)
    assert (qweight.dtype == qinput.dtype)
    assert (qbias is None or qbias.dtype == torch.int32)
    assert (isinstance(input_zero_point, int))
    assert (isinstance(output_zero_point, int))
    assert (isinstance(input_scale, float))
    assert (isinstance(output_scale, float))
    assert (weight_scale.dtype == torch.float)

    shifted_qbias = shift_quantized_linear_bias(qbias, qweight, input_zero_point)

    # Step 1: integer-based fully-connected (8-bit multiplication with 32-bit accumulation)
    output = torch.nn.functional.linear(
        qinput.to(torch.int32), qweight.to(torch.int32), shifted_qbias)

    # Step 2: scale the output
    folded_scale = (input_scale * weight_scale) / output_scale

    folded_int_scale, frac_bits = float_to_fixed_point(folded_scale, feature_bitwidth)
    folded_scale_int_reshaped = folded_int_scale.view(1, output.shape[1])
    frac_bits_reshaped = frac_bits.view(1, output.shape[1])
    output_scaled = (output * folded_scale_int_reshaped) >> frac_bits_reshaped

    # Step 3: shift output by output_zero_point
    output_shifted = output_scaled + output_zero_point

    output = output_shifted.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output


class QLinear(torch.nn.Module):
    def __init__(self, qweight, qbias,
                 input_zero_point, output_zero_point,
                 input_scale, weight_scale, output_scale,
                 feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # current version Pytorch does not support IntTensor as nn.Parameter
        self.register_buffer('qweight', qweight)
        self.register_buffer('qbias', qbias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        return quantized_linear(
            x, self.qweight, self.qbias,
            self.feature_bitwidth, self.weight_bitwidth,
            self.input_zero_point, self.output_zero_point,
            self.input_scale, self.weight_scale, self.output_scale
        )
