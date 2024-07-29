import torch


def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))

    return quantized_min, quantized_max


def get_quantization_scale_and_zero_point(fp_tensor, bitwidth):
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    r_min, r_max = fp_tensor.min().item(), fp_tensor.max().item()

    scale = (r_max - r_min) / (quantized_max - quantized_min)
    zero_point = quantized_min - (r_min / scale)

    if zero_point < quantized_min:
        zero_point = quantized_min
    elif zero_point > quantized_max:
        zero_point = quantized_max
    else:  # convert from float to int using round()
        zero_point = round(zero_point)

    return scale, int(zero_point)


def linear_quantize(fp_tensor, bitwidth, scale,
                    zero_point, dtype=torch.int8) -> torch.Tensor:
    fp_tensor_scaled = fp_tensor / scale
    shifted_tensor = torch.round(fp_tensor_scaled).to(torch.int32) + zero_point

    quantized_min, quantized_max = get_quantized_range(bitwidth)
    q_tensor = torch.clamp(shifted_tensor, quantized_min, quantized_max)

    return q_tensor.to(dtype)


def linear_quantize_feature(fp_tensor, bitwidth):
    scale, zero_point = get_quantization_scale_and_zero_point(fp_tensor, bitwidth)
    q_tensor = linear_quantize(fp_tensor, bitwidth, scale, zero_point)

    return q_tensor, scale, zero_point
