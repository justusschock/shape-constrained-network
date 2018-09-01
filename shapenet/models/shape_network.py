import torch
from .shape_layer import ShapeLayerPy, ShapeLayerCpp
from .feature_extractors import Img224x224Kernel7x7SeparatedDims
from .abstract_network import AbstractNetwork
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils import data
from shapenet.utils import CustomGroupNorm
import os
import numpy as np


class ShapeNetwork(AbstractNetwork):
    def __init__(self, eigen_shapes,
                 n_shape_params,
                 n_global_params,
                 n_translation_params=2,
                 img_size=224,
                 in_channels=1,
                 verbose=False,
                 norm_type='instance',
                 use_cpp=False,
                 **kwargs
                 ):
        """

        Parameters
        ----------
        eigen_shapes: np.ndarray
            eigen shapes (obtained py PCA)
        n_shape_params: int
            number of shape parameters
        n_global_params: int
            number of global scaling parameters
        n_translation_params: int
            number of translation parameters
        img_size: int
            image size
        in_channels: int
            number of input channels
        verbose: bool
            verbosity
        norm_type: string
            which normalization type to use. Must be one of ["instance", "batch", "group"]
        use_cpp: bool
            whether or not to use the C++ Implementation of the Shape Layer
        kwargs: dict
            additional keyword arguments
        """
        super().__init__()
        self._kwargs = kwargs
        self._n_shape_params = n_shape_params
        self._n_global_params = n_global_params
        self._n_translation_params = n_translation_params
        self._img_size = img_size
        self._verbose = verbose
        self._model = None
        self._shape_layer = None
        norm_dict = {'instance': torch.nn.InstanceNorm2d,
                     'batch': torch.nn.BatchNorm2d,
                     'group': CustomGroupNorm}
        norm_class = norm_dict.get(norm_type, None)
        args = [eigen_shapes, n_shape_params, n_global_params, n_translation_params, in_channels, norm_class, use_cpp]

        if img_size == 224:
            self._build_model_224(*args)
        else:
            raise NotImplementedError()

    def _build_model_224(self, eigen_shapes,
                         n_shape_params,
                         n_global_params,
                         n_translation_params,
                         in_channels,
                         norm_class,
                         use_cpp):
        """
        Builds model for image size 224

        Parameters
        ----------
        eigen_shapes: np.ndarray
            eigen shapes (obtained by PCA)
        n_shape_params: int
            number of shape parameters
        n_global_params: int
            number of global scaling parameters
        n_translation_params: int
            number of translation parameters
        in_channels: int
            number of input channels
        norm_class: Any
            class implementing a normalization
        use_cpp: bool
            whether or not to use the C++ implementation of the shape layer

        """

        model = Img224x224Kernel7x7SeparatedDims(in_channels,
                                                 n_shape_params+n_global_params+n_translation_params,
                                                 norm_class)

        shape_layer_cls = ShapeLayerCpp if use_cpp else ShapeLayerPy

        self._model = model
        self._shape_layer = shape_layer_cls(
            eigen_shapes,
            n_shape_params,
            n_global_params,
            224
        )

        self.register_buffer("indices_shape_params", torch.arange(0, n_shape_params).long())
        self.register_buffer("indices_translation_params",
                             torch.arange(n_shape_params,
                                          n_shape_params + n_translation_params).long())
        self.register_buffer("indices_global_params",
                             torch.arange(n_shape_params + n_translation_params,
                                          n_shape_params + n_translation_params + n_global_params).long())

    def forward(self, input_images):
        """
        Forward input batch through network and shape layer

        Parameters
        ----------
        input_images: torch.Tensor
            input batch

        Returns
        -------
        torch.Tensor predicted shapes
        """

        features = self._model(input_images)

        indices_shape_params = getattr(self, "indices_shape_params")
        indices_translation_params = getattr(self, "indices_translation_params")
        indices_global_params = getattr(self, "indices_global_params")
        shape_params = features.index_select(dim=1, index=indices_shape_params)
        translation_params = features.index_select(dim=1, index=indices_translation_params)
        global_params = features.index_select(dim=1, index=indices_global_params)

        shapes = self._shape_layer(shape_params, translation_params, global_params)
        return shapes

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module):
        self._model = model

    @property
    def n_params(self):
        return {"shape": self._n_shape_params,
                "global": self._n_global_params}

    @n_params.setter
    def n_params(self, number_dict: dict):
        for name, number in number_dict.items():
            setattr(self, "_%s_params" % name, number)

    @property
    def shape_layer(self):
        return self._shape_layer

    @shape_layer.setter
    def shape_layer(self, layer):
        raise AttributeError("cannot reset shape layer")

    @staticmethod
    def validate(dataloader: data.DataLoader, model, loss_fn, writer: SummaryWriter, **kwargs):
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
        device = kwargs.get("device", torch.device("cpu"))
        verbose = kwargs.get("verbose", True)
        out_dir = kwargs.get("out_dir", "./Results/UnknownEpoch")
        curr_epoch = kwargs.get("curr_epoch", -1)
        model = model.to(device).eval()

        if kwargs.get("save_outputs", False):
            os.makedirs(out_dir, exist_ok=True)

        if verbose:
            wrapper_fn = tqdm
        else:
            def linear_fn(x):
                return x
            wrapper_fn = linear_fn

        loss_vals = []
        for idx, data in enumerate(wrapper_fn(dataloader)):
            img, label = data[0].to(torch.float).to(device), data[1].to(torch.float).to(device)

            last_result = model(img)
            loss_value = loss_fn(last_result, label)

            loss_vals.append(np.asscalar(loss_value.detach().cpu().sum().numpy()))

        if verbose:
            print("\tMean Validation Loss:\t%f" % np.mean(np.asarray(loss_vals)))
        writer.add_scalar("Mean Validation Loss", np.asscalar(np.mean(np.asarray(loss_vals))), curr_epoch)
        return np.asscalar(np.mean(np.asarray(loss_vals)))

    @staticmethod
    def single_epoch(dataloader: data.DataLoader, optimizer: torch.optim.Optimizer, model, loss_fn,
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
        device = kwargs.get("device", torch.device("cpu"))

        verbose = kwargs.get("verbose", True)
        curr_epoch = kwargs.get("curr_epoch", -1)

        model = model.to(device).train()
        loss_vals = []

        if verbose:
            wrapper_fn = tqdm
        else:
            def linear_fn(x):
                return x
            wrapper_fn = linear_fn

        for idx, data in enumerate(wrapper_fn(dataloader)):
            img, label = data[0].to(torch.float).to(device), data[1].to(torch.float).to(device)

            last_result = model(img)
            loss_value = loss_fn(last_result, label)
            loss_vals.append(np.asscalar(loss_value.detach().cpu().sum().numpy()))

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        if verbose:
            print("\tMean Train Loss:\t%f" % np.mean(np.asarray(loss_vals)))
        writer.add_scalar("Mean Train Loss", np.asscalar(np.mean(np.asarray(loss_vals))), curr_epoch)

        return optimizer, model
