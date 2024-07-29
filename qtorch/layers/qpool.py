import torch


class QMaxPool2d(torch.nn.MaxPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based MaxPool
        return super().forward(x.float()).to(torch.int8)


class QAveragePool(torch.nn.MaxPool2d):
    def forward(self, x):
        # current version PyTorch does not support integer-based AveragePool
        return super().forward(x.float()).to(torch.int8)
