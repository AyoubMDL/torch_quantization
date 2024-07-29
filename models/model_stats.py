from torch import nn
from torchprofile import profile_macs


def get_model_flops(model, inputs):
    num_macs = profile_macs(model, inputs)
    return num_macs


def get_model_size(model: nn.Module, data_width=32):
    """
    calculate the model size in bits

    Args:
        model: Torch model
        data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()

    return num_elements * data_width
