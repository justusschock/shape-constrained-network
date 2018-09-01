import torch
from .abstract_network import AbstractFeatureExtractor


class Conv2dRelu(torch.nn.Module):
    """
    Block holding one Conv2d and one ReLU layer
    """
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        args: list
            positional arguments (passed to Conv2d)
        kwargs: dict
            keyword arguments (passed to Conv2d)
        """
        super().__init__()
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        self._relu = torch.nn.ReLU()

    def forward(self, input_batch):
        """
        Forward batch though layers

        Parameters
        ----------
        input_batch: torch.Tensor
            input batch

        Returns
        -------
        torch.Tensor: result
        """
        return self._relu(self._conv(input_batch))


class Conv3dRelu(torch.nn.Module):
    """
    Block holding one Conv3d and one ReLU layer
    """
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        args: list
            positional arguments (passed to Conv3d)
        kwargs: dict
            keyword arguments (passed to Conv3d)
        """
        super().__init__()
        self._conv = torch.nn.Conv3d(*args, **kwargs)
        self._relu = torch.nn.ReLU()

    def forward(self, input_batch):
        """
        Forward batch though layers

        Parameters
        ----------
        input_batch: torch.Tensor
            input batch

        Returns
        -------
        torch.Tensor: result
        """
        return self._relu(self._conv(input_batch))


class Img224x224Kernel3x3(AbstractFeatureExtractor):
    @staticmethod
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
        model = torch.nn.Sequential()
        model.add_module("conv_1_1", Conv2dRelu(in_channels, 64, 3, 1))
        model.add_module("conv_1_2", Conv2dRelu(64, 64, 3, 1))

        model.add_module("down_conv_1", Conv2dRelu(64, 64, 3, 2))
        if norm_class is not None:
            model.add_module("norm_1", norm_class(64))

        if p_dropout:
            model.add_module("dropout_1", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_2_1", Conv2dRelu(64, 128, 3, 1))
        model.add_module("conv_2_2", Conv2dRelu(128, 128, 3, 1))

        model.add_module("down_conv_2", Conv2dRelu(128, 128, 3, 2))
        if norm_class is not None:
            model.add_module("norm_2", norm_class(128))
        if p_dropout:
            model.add_module("dropout_2", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_3_1", Conv2dRelu(128, 256, 3, 1))
        model.add_module("conv_3_2", Conv2dRelu(256, 256, 3, 1))
        model.add_module("conv_3_3", Conv2dRelu(256, 256, 3, 1))
        model.add_module("conv_3_4", Conv2dRelu(256, 256, 3, 1))

        model.add_module("down_conv_3", Conv2dRelu(256, 256, 3, 2))
        if norm_class is not None:
            model.add_module("norm_3", norm_class(256))
        if p_dropout:
            model.add_module("dropout_3", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_4_1", Conv2dRelu(256, 512, 3, 1))
        model.add_module("conv_4_2", Conv2dRelu(512, 512, 3, 1))
        model.add_module("conv_4_3", Conv2dRelu(512, 512, 3, 1))
        model.add_module("conv_4_4", Conv2dRelu(512, 512, 3, 1))

        model.add_module("down_conv_4", Conv2dRelu(512, 256, 3, 2))
        if norm_class is not None:
            model.add_module("norm_4", norm_class(256))
        if p_dropout:
            model.add_module("dropout_4", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_5_1", Conv2dRelu(256, 128, 3, 1))
        model.add_module("conv_6_1", Conv2dRelu(128, 128, 3, 1))
        model.add_module("conv_7_1", torch.nn.Conv2d(128, out_features, 2, 1))

        return model


class Img224x224Kernel3x3SeparatedDims(AbstractFeatureExtractor):
    @staticmethod
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
        model = torch.nn.Sequential()
        model.add_module("conv_1_1", Conv2dRelu(in_channels, 64, 3, 1))
        model.add_module("conv_1_1_2", Conv2dRelu(64, 64, (1, 3), 1))
        model.add_module("conv_1_2_1", Conv2dRelu(64, 64, (3, 1), 1))
        model.add_module("conv_1_2_2", Conv2dRelu(64, 64, (1, 3), 1))

        model.add_module("down_conv_1", Conv2dRelu(64, 64, 3, 2))
        if norm_class is not None:
            model.add_module("norm_1", norm_class(64))
        if p_dropout:
            model.add_module("dropout_1", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_2_1_1", Conv2dRelu(64, 128, (3, 1), 1))
        model.add_module("conv_2_1_2", Conv2dRelu(128, 128, (1, 3), 1))
        model.add_module("conv_2_2_1", Conv2dRelu(128, 128, (3, 1), 1))
        model.add_module("conv_2_2_2", Conv2dRelu(128, 128, (1, 3), 1))

        model.add_module("down_conv_2", Conv2dRelu(128, 128, 3, 2))
        if norm_class is not None:
            model.add_module("norm_2", norm_class(128))
        if p_dropout:
            model.add_module("dropout_2", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_3_1_1", Conv2dRelu(128, 256, (3, 1), 1))
        model.add_module("conv_3_1_2", Conv2dRelu(256, 256, (1, 3), 1))
        model.add_module("conv_3_2_1", Conv2dRelu(256, 256, (3, 1), 1))
        model.add_module("conv_3_2_2", Conv2dRelu(256, 256, (1, 3), 1))
        model.add_module("conv_3_3_1", Conv2dRelu(256, 256, (3, 1), 1))
        model.add_module("conv_3_3_2", Conv2dRelu(256, 256, (1, 3), 1))
        model.add_module("conv_3_4_1", Conv2dRelu(256, 256, (3, 1), 1))
        model.add_module("conv_3_4_2", Conv2dRelu(256, 256, (1, 3), 1))

        model.add_module("down_conv_3", Conv2dRelu(256, 256, 3, 2))
        if norm_class is not None:
            model.add_module("norm_3", norm_class(256))
        if p_dropout:
            model.add_module("dropout_3", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_4_1_1", Conv2dRelu(256, 512, (3, 1), 1))
        model.add_module("conv_4_1_2", Conv2dRelu(512, 512, (1, 3), 1))
        model.add_module("conv_4_2_1", Conv2dRelu(512, 512, (3, 1), 1))
        model.add_module("conv_4_2_2", Conv2dRelu(512, 512, (1, 3), 1))
        model.add_module("conv_4_3_1", Conv2dRelu(512, 512, (3, 1), 1))
        model.add_module("conv_4_3_2", Conv2dRelu(512, 512, (1, 3), 1))
        model.add_module("conv_4_4_1", Conv2dRelu(512, 512, (3, 1), 1))
        model.add_module("conv_4_4_2", Conv2dRelu(512, 512, (1, 3), 1))

        model.add_module("down_conv_4", Conv2dRelu(512, 256, 3, 2))
        if norm_class is not None:
            model.add_module("norm_4", norm_class(256))
        if p_dropout:
            model.add_module("dropout_4", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_5_1_1", Conv2dRelu(256, 128, (3, 1), 1))
        model.add_module("conv_5_1_2", Conv2dRelu(128, 128, (1, 3), 1))
        model.add_module("conv_6_1_1", Conv2dRelu(128, 128, (3, 1), 1))
        model.add_module("conv_6_1_2", Conv2dRelu(128, 128, (1, 3), 1))
        model.add_module("conv_7_1", torch.nn.Conv2d(128, out_features, 2, 1))

        return model


class Img224x224Kernel7x7SeparatedDims(AbstractFeatureExtractor):
    @staticmethod
    def _build_model(in_channels, out_params, norm_class, p_dropout):
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
        model = torch.nn.Sequential()

        model.add_module("conv_1", Conv2dRelu(in_channels, 64, (7, 1)))
        model.add_module("conv_2", Conv2dRelu(64, 64, (1, 7)))

        model.add_module("down_conv_1", Conv2dRelu(64, 128, (7, 7), stride=2))
        if norm_class is not None:
            model.add_module("norm_1", norm_class(128))
        if p_dropout:
            model.add_module("dropout_1", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_3", Conv2dRelu(128, 128, (7, 1)))
        model.add_module("conv_4", Conv2dRelu(128, 128, (1, 7)))

        model.add_module("down_conv_2", Conv2dRelu(128, 256, (7, 7), stride=2))
        if norm_class is not None:
            model.add_module("norm_2", norm_class(256))
        if p_dropout:
            model.add_module("dropout_2", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_5", Conv2dRelu(256, 256, (5, 1)))
        model.add_module("conv_6", Conv2dRelu(256, 256, (1, 5)))

        model.add_module("down_conv_3", Conv2dRelu(256, 256, (5, 5), stride=2))
        if norm_class is not None:
            model.add_module("norm_3", norm_class(256))
        if p_dropout:
            model.add_module("dropout_3", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_7", Conv2dRelu(256, 256, (5, 1)))
        model.add_module("conv_8", Conv2dRelu(256, 256, (1, 5)))

        model.add_module("down_conv_4", Conv2dRelu(256, 128, (5, 5), stride=2))
        if norm_class is not None:
            model.add_module("norm_4", norm_class(128))
        if p_dropout:
            model.add_module("dropout_4", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_9", Conv2dRelu(128, 128, (3, 1)))
        model.add_module("conv_10", Conv2dRelu(128, 128, (1, 3)))
        model.add_module("conv_11", Conv2dRelu(128, 128, (3, 1)))
        model.add_module("conv_12", Conv2dRelu(128, 128, (1, 3)))

        model.add_module("final_conv", torch.nn.Conv2d(128, out_params, (2, 2)))

        return model


class Img1024x1024Kernel9x9SeparatedDims(AbstractFeatureExtractor):
    @staticmethod
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
        model = torch.nn.Sequential()

        model.add_module("conv_1", Conv2dRelu(in_channels, 64, (9, 1)))
        model.add_module("conv_2", Conv2dRelu(64, 64, (1, 9)))

        model.add_module("down_conv_1", Conv2dRelu(64, 128, (9, 9), stride=3))
        if norm_class is not None:
            model.add_module("norm_1", norm_class(128))
        if p_dropout:
            model.add_module("dropout_1", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_3", Conv2dRelu(128, 128, (9, 1)))
        model.add_module("conv_4", Conv2dRelu(128, 128, (1, 9)))

        model.add_module("down_conv_2", Conv2dRelu(128, 256, (7, 7), stride=3))
        if norm_class is not None:
            model.add_module("norm_2", norm_class(256))
        if p_dropout:
            model.add_module("dropout_2", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_5", Conv2dRelu(256, 256, (7, 1)))
        model.add_module("conv_6", Conv2dRelu(256, 256, (1, 7)))

        model.add_module("down_conv_3", Conv2dRelu(256, 256, (5, 5), stride=3))
        if norm_class is not None:
            model.add_module("norm_3", norm_class(256))
        if p_dropout:
            model.add_module("dropout_3", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_7", Conv2dRelu(256, 256, (5, 1)))
        model.add_module("conv_8", Conv2dRelu(256, 256, (1, 5)))

        model.add_module("down_conv_4", Conv2dRelu(256, 128, (5, 5), stride=2))
        if norm_class is not None:
            model.add_module("norm_4", norm_class(128))
        if p_dropout:
            model.add_module("dropout_4", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_9", Conv2dRelu(128, 128, (3, 1)))
        model.add_module("conv_10", Conv2dRelu(128, 128, (1, 3)))

        model.add_module("down_conv_5", Conv2dRelu(128, 128, (3, 3), stride=2))

        model.add_module("conv_11", Conv2dRelu(128, 128, (3, 1)))
        model.add_module("conv_12", Conv2dRelu(128, 128, (1, 3)))

        model.add_module("final_conv", torch.nn.Conv2d(128, out_features, (3, 3)))

        return model

    class Img512x512Kernel9x9SeparatedDims(AbstractFeatureExtractor):
        @staticmethod
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
            model = torch.nn.Sequential()

            model.add_module("conv_1", Conv2dRelu(in_channels, 64, (9, 1)))
            model.add_module("conv_2", Conv2dRelu(64, 64, (1, 9)))

            model.add_module("down_conv_1", Conv2dRelu(64, 128, (9, 9), stride=3))
            if norm_class is not None:
                model.add_module("norm_1", norm_class(128))
            if p_dropout:
                model.add_module("dropout_1", torch.nn.Dropout2d(p_dropout))

            model.add_module("conv_3", Conv2dRelu(128, 128, (9, 1)))
            model.add_module("conv_4", Conv2dRelu(128, 128, (1, 9)))

            model.add_module("down_conv_2", Conv2dRelu(128, 256, (7, 7), stride=3))
            if norm_class is not None:
                model.add_module("norm_2", norm_class(256))
            if p_dropout:
                model.add_module("dropout_2", torch.nn.Dropout2d(p_dropout))

            model.add_module("conv_5", Conv2dRelu(256, 256, (7, 1)))
            model.add_module("conv_6", Conv2dRelu(256, 256, (1, 7)))

            model.add_module("down_conv_3", Conv2dRelu(256, 256, (5, 5), stride=3))
            if norm_class is not None:
                model.add_module("norm_3", norm_class(256))
            if p_dropout:
                model.add_module("dropout_3", torch.nn.Dropout2d(p_dropout))

            model.add_module("conv_7", Conv2dRelu(256, 256, (5, 1)))
            model.add_module("conv_8", Conv2dRelu(256, 256, (1, 5)))

            model.add_module("down_conv_4", Conv2dRelu(256, 128, (5, 5), stride=2))
            if norm_class is not None:
                model.add_module("norm_4", norm_class(128))
            if p_dropout:
                model.add_module("dropout_4", torch.nn.Dropout2d(p_dropout))

            model.add_module("conv_9", Conv2dRelu(128, 128, (3, 1)))
            model.add_module("conv_10", Conv2dRelu(128, 128, (1, 3)))

            model.add_module("down_conv_5", Conv2dRelu(128, 128, (3, 3), stride=2))

            model.add_module("conv_11", Conv2dRelu(128, 128, (3, 1)))
            model.add_module("conv_12", Conv2dRelu(128, 128, (1, 3)))

            model.add_module("final_conv", torch.nn.Conv2d(128, out_features, (3, 3)))

            return model





