import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Union, List, Dict

import logging
logging.basicConfig(level=logging.INFO)


class ActivationRegistry:
    def __init__(self):
        # Initialize the mapping for activation functions
        self.func_map: Dict[str, Callable] = {}

        # Register default activation functions
        self.register('tanh', nn.Tanh)
        self.register('relu', nn.ReLU)
        self.register('silu', nn.SiLU)
        self.register('mish', nn.Mish)
        self.register('sigmoid', nn.Sigmoid)
        self.register('softmax', lambda dim=1: nn.Softmax(dim=dim))
        self.register('log_softmax', lambda dim=1: nn.LogSoftmax(dim=dim))

    def register(self, name: str, func: Callable):
        if name in self.func_map:
            logging.info(f'Activation function "{name}" is already registered. Override it.')
        
        self.func_map[name] = func

    def get(self, name: str, **kwargs) -> Callable:
        if name not in self.func_map:
            raise KeyError(f'Activation function "{name}" is not registered.')

        # If the function is parameterized (e.g., softmax), allow dynamic configuration
        func = self.func_map[name]
        if callable(func):
            return func(**kwargs) if kwargs else func()
        return func

    def list_registered(self) -> List[str]:
        return list(self.func_map.keys())

activation_registry = ActivationRegistry()


class MLP(nn.Module):
    def __init__(
        self,
        features: list,
        hid_trans: str = 'mish',
        out_trans: Union[str, bool] = False,
        norm: Union[str, bool] = False,
        hid_norm: Union[str, bool] = False,
        drop: Union[float, bool] = False,
        hid_drop: Union[float, bool] = False,
    ):
        super(MLP, self).__init__()
        assert len(features) > 1, 'MLP must have at least 2 layers (input and output)!'

        # Apply global normalization and dropout if specified
        if norm:
            hid_norm = out_norm = norm
        else:
            out_norm = False
        if drop:
            hid_drop = out_drop = drop
        else:
            out_drop = False

        # Build the MLP layers
        layers = []
        for i in range(1, len(features)):
            layers.append(nn.Linear(features[i - 1], features[i]))
            if i < len(features) - 1:  # Hidden layers
                layers.append(Layer1D(features[i], hid_norm, hid_trans, hid_drop))
            else:  # Output layer
                layers.append(Layer1D(features[i], out_norm, out_trans, out_drop))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Layer1D(nn.Module):
    def __init__(
        self,
        dim: Union[int, bool] = False,
        norm: Union[str, bool] = False,
        trans: Union[str, bool] = False,
        drop: Union[float, bool] = False,
    ):
        super(Layer1D, self).__init__()
        layers = []

        # Add normalization layer
        if norm == 'bn':
            layers.append(nn.BatchNorm1d(dim))
        elif norm == 'ln':
            layers.append(nn.LayerNorm(dim))

        # Add activation function
        if trans:
            layers.append(activation_registry.get(trans))

        # Add dropout layer
        if drop:
            layers.append(nn.Dropout(drop))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


