import torch


def float_to_fixed_point(x, bitwidth, signed=True, clamp=False):
    """Converts a number to a FixedPoint representation

    Args:
        x (torch.Tensor): the source number or tensor
        bitwidth (int): the desired bitwidth
        signed (bool, optional): when reserving a bit for the sign. Defaults to True.
        clamp (bool, optional): clamp the resulted fractional bits to mantissa_bits.
            Defaults to False.

    Returns:
        torch.Tensor, torch.Tensor: the mantissa and fractional bits
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if torch.any(torch.isinf(x)):
        raise ValueError(f"Infinite values are not supported. Receives: {x}")

    # Evaluate the number of bits available for the mantissa
    mantissa_bits = bitwidth - 1 if signed else bitwidth

    # Evaluate the number of bits required to represent the whole part of x
    y = torch.abs(torch.where(x == 0, torch.tensor(1.0, dtype=x.dtype), x))
    whole_bits = torch.ceil(torch.log2(y)).to(torch.int32)

    # Deduce the number of bits required for the fractional part of x
    frac_bits = mantissa_bits - whole_bits
    if clamp:
        frac_bits = torch.minimum(frac_bits, mantissa_bits)

    # Evaluate the 'scale', which is the smallest value that can be represented (as 1)
    scale = 2. ** -frac_bits

    # Evaluate the minimum and maximum values for the mantissa
    mantissa_min = -2 ** mantissa_bits if signed else 0
    mantissa_max = 2 ** mantissa_bits - 1

    # Evaluate the mantissa by quantizing x with the scale, clipping to the min and max
    mantissa = torch.clamp(torch.round(x / scale), mantissa_min, mantissa_max).to(torch.int32)

    return mantissa, frac_bits
