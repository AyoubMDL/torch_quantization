import copy

import torch


def _fuse_conv_bn(conv, bn):
    assert conv.bias is None

    factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
    conv.bias = torch.nn.Parameter(- bn.running_mean.data * factor + bn.bias.data)

    return conv


def fuse_batchnorms(model):
    model_fused = copy.deepcopy(model)
    fused_backbone = []
    ptr = 0

    while ptr < len(model_fused.backbone):
        if isinstance(model_fused.backbone[ptr], torch.nn.Conv2d) and \
                isinstance(model_fused.backbone[ptr + 1], torch.nn.BatchNorm2d):
            fused_backbone.append(_fuse_conv_bn(
                model_fused.backbone[ptr], model_fused.backbone[ptr + 1]))
            ptr += 2
        else:
            fused_backbone.append(model_fused.backbone[ptr])
            ptr += 1
    model_fused.backbone = torch.nn.Sequential(*fused_backbone)

    # sanity check, no BN anymore
    for m in model_fused.modules():
        assert not isinstance(m, torch.nn.BatchNorm2d)

    return model_fused
