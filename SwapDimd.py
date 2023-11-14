#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:52:47 2023

@author: sun
"""


from __future__ import annotations
from typing import Hashable, Mapping
from monai.transforms.utils_pytorch_numpy_unification import moveaxis
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform

from torch import squeeze


class SwapDimd(MapTransform):

    def __init__(self, keys: KeysCollection, depth_dim: int = -1, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.depth_dim = depth_dim

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = squeeze(moveaxis(d[key], self.depth_dim, 1))
        return d