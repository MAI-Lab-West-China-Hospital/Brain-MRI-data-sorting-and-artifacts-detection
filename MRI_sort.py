#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:21:28 2023

@author: sun
"""

import os
import shutil
import fnmatch
import torch
import pandas as pd
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityd, EnsureChannelFirstd
from MRI_models import MRIEfficientNet
from MRI_transforms import ExtractSlicesd, SwapDimd
import argparse


def dataset_from_folder(data_root_path):
    dataset=list()
    for path,dirs,files in os.walk(data_root_path):
        for f in fnmatch.filter(files, '*.nii.gz'):
            dataset.append({'img':os.path.join(path,f)})    
    return dataset


def MRIsort(input_path, output_path):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    pred_set=dataset_from_folder(input_path)
    
    class_file = pd.read_csv("./pretrained_model/labelmap.csv", index_col=0, header=None)
    classes = class_file.index.tolist()
    
    for i in classes:
        if not os.path.exists(os.path.join(output_path, i)):
            os.makedirs(os.path.join(output_path, i))
    
    num_slices = 3
    gap = 2

    
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
    
    pred_ds = Dataset(pred_set, transform=transforms)
    pred_loader = DataLoader(pred_ds, batch_size=56, num_workers=4, pin_memory=torch.cuda.is_available())
    
    device= torch.device("cuda:0")
    
    model = MRIEfficientNet("efficientnet-b0", spatial_dims=2, in_channels=num_slices,
                               num_classes=len(classes), pretrained=False, dropout_rate=0.2).to(device)
    
    weights = torch.load("./pretrained_model/best_model.pth")
    model.load_state_dict(weights)
    
    with torch.no_grad():
        for pred_batch in pred_loader:
            pred_images = pred_batch["img"].to(device)
            pred_outputs = model(pred_images).argmax(dim=1)
            for i in range(len(pred_outputs)):
                class_folder = classes[pred_outputs[i]]
                src=pred_batch["img"][i].meta["filename_or_obj"]
                dist=os.path.join(output_path, class_folder)
                shutil.copy(src, dist)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', help='Input NIFTIs path', required=True)
    parser.add_argument('--output', help='Path to output sorted NIFTIs', default='./output')
    
    args = parser.parse_args()
    
    MRIsort(args.input, args.output)
    
if __name__ == '__main__':
    main()
    
    

