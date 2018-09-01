import torch
from torch.utils.data import DataLoader
from abc import abstractmethod
from tensorboardX import SummaryWriter


class AbstractNetwork(torch.nn.Module):
    """
    Abstract Network class all other networks should be derived from
    """

    @abstractmethod
    def __init__(self):
        """
        Abstract Init function
        """
        super().__init__()
        pass

    @abstractmethod
    def forward(self, *inputs):
        """
        Defines forward through network

        Parameters
        ----------
        inputs: list
            network inputs
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate(dataloader: DataLoader, model, loss_fn, writer: SummaryWriter, **kwargs):
        """
        Validate on dataset

        Parameters
        ----------
        dataloader: DataLoader
            loader for validation data
        model: AbstractNetwork
            model to validate
        loss_fn: torch.nn.Module
            loss function to monitor
        writer: SummaryWriter
            Summarywriter to log loss values
        kwargs: dict
            additional keyword arguments
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def single_epoch(dataloader: DataLoader, optimizer: torch.optim.Optimizer, model, loss_fn,
                     writer: SummaryWriter, **kwargs):
        """
        Train single epoch

        Parameters
        ----------
        dataloader: DataLoader
            loader for training data
        optimizer: torch.optim.Optimizer
            optimizer implementing the actual optimization algorithm
        model: AbstractNetwork
            model to train
        loss_fn: torch.nn.Module
            loss function to monitor and calculate gradients
        writer: SummaryWriter
            Summarywriter to log loss values
        kwargs: dict
            additional keyword arguments
        """
        raise NotImplementedError


class AbstractFeatureExtractor(torch.nn.Module):
    """
    Abstract Feature Extractor Class all further feature extracotrs should be derived from
    """
    def __init__(self, in_channels, out_params, norm_class, p_dropout=0):
        """

        Parameters
        ----------
        in_channels: int
            number of input channels
        out_params: int
            number of outputs
        norm_class: Any
            Class implementing a normalization
        p_dropout: float
            dropout probability
        """
        super().__init__()
        self.model = self._build_model(in_channels, out_params, norm_class, p_dropout)

    def forward(self, input_batch):
        """
        Feed batch through network

        Parameters
        ----------
        input_batch: torch.Tensor
            batch to feed through network

        Returns
        -------
        torch.Tensor exracted features
        """
        return self.model(input_batch)

    @staticmethod
    @abstractmethod
    def _build_model(in_channels, out_features, norm_class, p_dropout):
        """
        Build the actual model structure

        Parameters
        ----------
        in_channels: int
            number of input channels
        out_features: int
            number of outputs
        norm_class: Any
            class implementing a normalization
        p_dropout: float
            dropout probability

        Returns
        -------
        torch.nn.Module: ensembled model
        """
        raise NotImplementedError
