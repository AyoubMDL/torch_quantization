import functools

import torch


def calibrate(model, samples):
    input_activation = {}
    output_activation = {}

    def _record_range(self, x, y, module_name):
        x = x[0]
        input_activation[module_name] = x.detach()
        output_activation[module_name] = y.detach()

    all_hooks = []

    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear,
                          torch.nn.ReLU)):
            all_hooks.append(m.register_forward_hook(
                functools.partial(_record_range, module_name=name)
            ))

    model(samples.cpu())

    for h in all_hooks:
        h.remove()

    return input_activation, output_activation
