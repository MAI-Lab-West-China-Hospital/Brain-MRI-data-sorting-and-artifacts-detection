#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:42:17 2023

@author: hmrrc
"""

from __future__ import annotations

from typing import Hashable, Mapping

import numpy as np

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta

from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils import convert_to_tensor
from monai.utils.enums import PostFix


# from monai.transforms import *
# import torch
# from SwapDimd import SwapDimd
# from monai.data import DataLoader, DistributedSampler, CacheDataset, Dataset, PersistentDataset
__all__ = [
    "ExtractSlicesd"
    ]
    
DEFAULT_POST_FIX = PostFix.meta()

class ExtractSlicesd(RandomizableTransform, MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        num_slices: int,
        gap: int,
        prob: float = 1,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.num_slices = num_slices
        self.gap = gap

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> ExtractSlicesd:
        self._rand_state = self.R
        return super().set_random_state(seed, state)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # Expect all the specified keys have the same spatial shape
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        spatial_shape = d[first_key].shape[1:]
        center_slice = spatial_shape[2] // 2
        #print(center_slice)

        # Create a copy of the input data
        result_data = dict(data)

        for key in self.key_iterator(d): 
            channel_data = d[key]
            
            start_index = center_slice - ((self.num_slices-1) // 2 * (self.gap + 1))
            end_index = center_slice + ((self.num_slices-1) // 2 * (self.gap + 1))
            
            if start_index > -1:
                result_data[key] = channel_data[:, :, :, start_index:end_index + 1:(self.gap + 1)]
            
        return result_data
