from models.device import DEVICE
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model: nn.Module,
          data_loader: DataLoader,
          criterion: nn.Module,
          optimizer: Optimizer,
          scheduler: LambdaLR,
          callbacks=None) -> None:
    model.train()

    for inputs, targets in tqdm(data_loader, desc="train", leave=False):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()
        scheduler.step()

        if callbacks is not None:
            for callback in callbacks:
                callback()
