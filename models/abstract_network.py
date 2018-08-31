import torch
import data
from torch.utils.data import DataLoader
from abc import abstractmethod
from tensorboardX import SummaryWriter


class AbstractNetwork(torch.nn.Module):

    @abstractmethod
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate(dataloader: DataLoader, model, loss_fn, writer: SummaryWriter, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def single_epoch(dataloader: DataLoader, optimizer: torch.optim.Optimizer, model, loss_fn,
                     writer: SummaryWriter, **kwargs):
        raise NotImplementedError


class AbstractFeatureExtractor(torch.nn.Module):
    def __init__(self, in_channels, out_params, norm_class, p_dropout=0):
        super().__init__()
        self.model = self._build_model(in_channels, out_params, norm_class, p_dropout)

    def forward(self, input_batch):
        return self.model(input_batch)

    @staticmethod
    @abstractmethod
    def _build_model(in_channels, out_features, norm_class, p_dropout):
        raise NotImplementedError
