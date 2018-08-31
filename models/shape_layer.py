import numpy as np
import torch
import torch.nn.functional as F
import warnings
from torch.utils.cpp_extension import load as load_cpp


class ShapeLayerPy(torch.nn.Module):
    def __init__(self, shapes, n_shape_params: int, n_global_params: int, img_size: int):
        super().__init__()

        self.register_buffer("_shape_mean", torch.from_numpy(shapes[0]).float().unsqueeze(0))
        components = []
        for i, _shape in enumerate(shapes[1:]):
            components.append(torch.from_numpy(_shape).float().unsqueeze(0))

        component_tensor = torch.cat(components).unsqueeze(0)
        self.register_buffer("_shape_components", component_tensor)
        self._n_shape_params = n_shape_params
        self._n_global_params = n_global_params
        self._img_size = img_size

    def forward(self, shape_params: torch.Tensor, translation_params: torch.Tensor, global_params: torch.Tensor):
        shapes = getattr(self, "_shape_mean").clone()
        shapes = shapes.expand(shape_params.size(0), *shapes.size()[1:])
        components = getattr(self, "_shape_components")
        components = components.expand(shape_params.size(0), *components.size()[1:])
        weighted_components = components.mul(shape_params.expand_as(components))

        translation_params = translation_params.squeeze(-1).squeeze(-1).unsqueeze(1)
        shapes = shapes.add(weighted_components.sum(dim=1)).add(translation_params * self._img_size)
        shapes = shapes.mul(global_params.squeeze(-1).squeeze(-1).unsqueeze(1).expand_as(shapes))
        return shapes

    @property
    def shape_mean(self):
        return getattr(self, "_shape_mean")

    @shape_mean.setter
    def shape_mean(self, mean):
        setattr(self, "_shape_mean", mean)

    @property
    def n_params(self):
        return {"shape": self._n_shape_params,
                "global": self._n_global_params}

    @n_params.setter
    def n_params(self, n_param_dict: dict):
        for name, number in n_param_dict.items():
            setattr(self, "_%s_params" % name, number)


class ShapeLayerCpp(torch.nn.Module):
    def __init__(self, shapes, n_shape_params: int, n_global_params: int, img_size: int,
                 verbose: True):
        super().__init__()

        self.register_buffer("_shape_mean", torch.from_numpy(shapes[0]).float().unsqueeze(0))
        components = []
        for i, _shape in enumerate(shapes[1:]):
            components.append(torch.from_numpy(_shape).float().unsqueeze(0))

        component_tensor = torch.cat(components).unsqueeze(0)
        self.register_buffer("_shape_components", component_tensor)
        self._n_shape_params = n_shape_params
        self._n_global_params = n_global_params
        self._img_size = img_size
        self._func = load_cpp("shape_function", sources=["shape_layer.cpp"], verbose=verbose)

    def forward(self, shape_params: torch.Tensor, translation_params: torch.Tensor, global_params: torch.Tensor):

        shapes = self._func.forward(shape_params,
                                    translation_params,
                                    global_params,
                                    getattr(self, "_shape_mean"),
                                    getattr(self, "_shape_components"),
                                    self._n_shape_params,
                                    self._n_global_params,
                                    self._img_size)

        return shapes

    @property
    def shape_mean(self):
        return getattr(self, "_shape_mean")

    @shape_mean.setter
    def shape_mean(self, mean):
        setattr(self, "_shape_mean", mean)

    @property
    def n_params(self):
        return {"shape": self._n_shape_params,
                "global": self._n_global_params}

    @n_params.setter
    def n_params(self, n_param_dict: dict):
        for name, number in n_param_dict.items():
            setattr(self, "_%s_params" % name, number)


