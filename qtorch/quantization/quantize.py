import copy

import torch

from ..layers import QAveragePool, QConv2d, QLinear, QMaxPool2d
from .bias import linear_quantize_bias_per_output_channel
from .linear_quantize import get_quantization_scale_and_zero_point
from .weight_quantization import linear_quantize_weight_per_channel
from .fuse_bn import fuse_batchnorms
from .calibrate import calibrate


def quantize(model, feature_bitwidth, weight_bitwidth, sample_data):
    quantized_model = copy.deepcopy(model)
    quantized_model = fuse_batchnorms(quantized_model)

    input_activation, output_activation = calibrate(quantized_model, sample_data)

    quantized_backbone = []
    ptr = 0

    while ptr < len(quantized_model.backbone):
        if isinstance(quantized_model.backbone[ptr], torch.nn.Conv2d) and \
                isinstance(quantized_model.backbone[ptr + 1], torch.nn.ReLU):
            conv = quantized_model.backbone[ptr]
            conv_name = f'backbone.{ptr}'
            relu_name = f'backbone.{ptr + 1}'

            input_scale, input_zero_point = get_quantization_scale_and_zero_point(
                input_activation[conv_name], feature_bitwidth
            )

            output_scale, output_zero_point = get_quantization_scale_and_zero_point(
                output_activation[relu_name], feature_bitwidth
            )

            quantized_weight, weight_scale, weight_zero_point = \
                linear_quantize_weight_per_channel(conv.weight.data, weight_bitwidth)

            quantized_bias, bias_scale, bias_zero_point = \
                linear_quantize_bias_per_output_channel(
                    conv.bias.data, weight_scale, input_scale)

            quantized_conv = QConv2d(
                quantized_weight, quantized_bias,
                input_zero_point, output_zero_point,
                input_scale, weight_scale, output_scale,
                conv.stride, conv.padding, conv.dilation, conv.groups,
                feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
            )

            quantized_backbone.append(quantized_conv)
            ptr += 2

        elif isinstance(quantized_model.backbone[ptr], torch.nn.MaxPool2d):
            quantized_backbone.append(QMaxPool2d(
                kernel_size=quantized_model.backbone[ptr].kernel_size,
                stride=quantized_model.backbone[ptr].stride
            ))
            ptr += 1

        elif isinstance(quantized_model.backbone[ptr], torch.nn.AvgPool2d):
            quantized_backbone.append(QAveragePool(
                kernel_size=quantized_model.backbone[ptr].kernel_size,
                stride=quantized_model.backbone[ptr].stride
            ))
            ptr += 1
        else:
            raise NotImplementedError(type(quantized_model.backbone[ptr]))

    quantized_model.backbone = torch.nn.Sequential(*quantized_backbone)

    # quantizing the classifier
    fc_name = 'classifier'
    fc = model.classifier

    input_scale, input_zero_point = get_quantization_scale_and_zero_point(
        input_activation[fc_name], feature_bitwidth)

    output_scale, output_zero_point = get_quantization_scale_and_zero_point(
        output_activation[fc_name], feature_bitwidth)

    quantized_weight, weight_scale, _ = linear_quantize_weight_per_channel(fc.weight.data,
                                                                           weight_bitwidth)

    quantized_bias, _, _ = linear_quantize_bias_per_output_channel(fc.bias.data,
                                                                   weight_scale,
                                                                   input_scale)

    quantized_model.classifier = QLinear(
        quantized_weight, quantized_bias,
        input_zero_point, output_zero_point,
        input_scale, weight_scale, output_scale,
        feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
    )

    return quantized_model
