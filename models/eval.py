import torch
from models.device import DEVICE
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.inference_mode()
def evaluate(model: nn.Module,
             data_loader: DataLoader,
             extra_preprocess=None) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(data_loader, desc="eval", leave=False):
        inputs = inputs.to(DEVICE)
        if extra_preprocess is not None:
            for preprocess in extra_preprocess:
                inputs = preprocess(inputs)

        targets = targets.to(DEVICE)

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()
