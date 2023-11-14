#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:21:28 2023

@author: sun
"""
import io

import os
import shutil
import fnmatch
import torch
import pandas as pd
import monai
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityd, EnsureChannelFirstd
from collections import OrderedDict
from EfficientNet import MyEfficientNet
from ExtractSlicesd import ExtractSlicesd
from SwapDimd import SwapDimd

num_slices = 3
gap = 2

def dataset_from_folder(data_root_path):
    dataset=list()
    for path,dirs,files in os.walk(data_root_path):
        for f in fnmatch.filter(files, '*.nii.gz'):
            dataset.append({'img':os.path.join(path,f)})    
    return dataset

val_path='../NII'
output_path='../SORT/'
val_set=dataset_from_folder(val_path)

class_file = pd.read_csv("./pretrained_model/labelmap.csv", index_col=0, header=None)
classes = class_file.index.tolist()
class_to_idx = class_file.to_dict()[1]

for i in classes:
    if not os.path.exists(os.path.join(output_path, i)):
        os.makedirs(os.path.join(output_path, i))


transforms = Compose(
        [
            LoadImaged(keys=['img']),
            EnsureChannelFirstd(keys=['img']),
            Resized(keys=["img"], spatial_size=(128,128,21)),
            ScaleIntensityd(keys=['img']), 
            ExtractSlicesd(keys=['img'], num_slices=num_slices , gap=gap), 
            SwapDimd(keys=['img'])
        ]
    )

val_ds = Dataset(val_set, transform=transforms)
val_loader = DataLoader(val_ds, batch_size=56, num_workers=4, pin_memory=torch.cuda.is_available())

device= torch.device("cuda:0")

model = MyEfficientNet("efficientnet-b0", spatial_dims=2, in_channels=num_slices,
                           num_classes=len(classes), pretrained=False, dropout_rate=0.2).to(device)

weights = torch.load("./pretrained_model/best_model.pth", map_location='cpu')

buffer = io.BytesIO()
torch.save(weights, buffer)
buffer.seek(0)

model.load_state_dict(torch.load(buffer))

with torch.no_grad():
    for val_batch in val_loader:
        val_images = val_batch["img"].to(device)
        val_outputs = model(val_images).argmax(dim=1)
        for i in range(len(val_outputs)):
            class_folder = classes[val_outputs[i]]
            src=val_batch["img"][i].meta["filename_or_obj"]
            dist=os.path.join(output_path, class_folder)
            shutil.copy(src, dist)

