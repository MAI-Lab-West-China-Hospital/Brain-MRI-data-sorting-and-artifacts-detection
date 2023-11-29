#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:05:00 2023

@author: sun
"""
import os
import sys
import fnmatch
from monai.data import DataLoader, DistributedSampler, PersistentDataset
from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityd, EnsureChannelFirstd
import pandas as pd
import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from MRI_models import MRIEfficientNet
from MRI_transforms import ExtractSlicesd, SwapDimd

def dataset_from_folder(data_root_path,labelmap):
    dataset=list()
    for path,dirs,files in os.walk(data_root_path):
        for f in fnmatch.filter(files, '*.nii.gz'):
            index = labelmap.index[labelmap[0]==os.path.basename(path)].tolist()
            dataset.append({'img':os.path.join(path,f),'label':labelmap.iloc[index[0],1]})    
    return dataset

def trainer(args):
    
    if args.local_rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    dist.init_process_group(backend="nccl",init_method="env://")
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    

    label_map=pd.read_csv(args.label_file, header=None)
    classes = label_map.index.tolist()
    

    train_files = dataset_from_folder(args.train_path, label_map)
    test_files = dataset_from_folder(args.test_path, label_map)

    transforms = Compose(
        [
            LoadImaged(keys=['img']),
            EnsureChannelFirstd(keys=['img']),
            Resized(keys=["img"], spatial_size=(64,64,64)),
            ScaleIntensityd(keys=['img']), 
            ExtractSlicesd(keys=['img'], num_slices=args.num_slices, gap=args.gap), 
            SwapDimd(keys=['img'])
        ]
    )
    
    
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    train_ds = PersistentDataset(train_files, transform=transforms, cache_dir=args.cache_dir)
    train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True) 
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers = 8, pin_memory=torch.cuda.is_available(), sampler=train_sampler)    
    
    test_ds = PersistentDataset(test_files, transform=transforms, cache_dir=args.cache_dir)
    test_sampler = DistributedSampler(dataset=test_ds, even_divisible=True, shuffle=True) 
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers = 8, pin_memory=torch.cuda.is_available(), sampler=test_sampler)    
    
    model = MRIEfficientNet("efficientnet-b0", spatial_dims=2, in_channels=args.num_slices, num_classes=len(classes), pretrained=False, dropout_rate=0.2).to(device)
    model = DistributedDataParallel(model, device_ids=[device])
    
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    
    epoch_len = len(train_ds)//train_loader.batch_size + 1
    
    val_interval = 5
    best_metric = -1
    best_metric_epoch = -1
    
    for epoch in range(args.num_epoch):
        print('-' * 10)
        print(f"epoch {epoch+1}/{args.num_epoch}")
        
        model.train()
        epoch_loss = 0
        step = 0
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
                
        for batch in train_loader:
            step += 1   
            inputs, labels = batch['img'].to(device),batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                
                for val_batch in test_loader:
                    val_images, val_labels = val_batch['img'].to(device),val_batch['label'].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)],dim=0)
                    y = torch.cat([y, val_labels],dim=0)
                    
                acc_value = torch.eq(y_pred.argmax(dim=1),y)
                acc_metric = acc_value.sum().item()/len(acc_value)
                
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model.pth")
                    print("saved new best metric model")
                
                print("current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(epoch + 1, acc_metric, best_metric, best_metric_epoch))
                
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), "final_model.pth")
        print("saved final model")
                
    dist.destroy_process_group()        
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, help="directory of train dataset")
    parser.add_argument("--test_path", type=str, help="directory of test dataset")
    parser.add_argument("--label_file", type=str, help="Label map file")
    parser.add_argument("--cachedir",default="./cache", type=str, help="directory of persistent cache")
    parser.add_argument("--batch_size", default=8, type=int, help="Size of mini batch")
    parser.add_argument("--num_slices", default=3, type=int, help="Number of slices extracted")
    parser.add_argument("--gap", default=2, type=int, help="Interval for slice extraction")
    parser.add_argument("--local-rank", type=int, help="node rank for distributed training")
    parser.add_argument("--num_epoch", default=300, type=int, metavar="N", help="number of total epochs to run")
    
    args = parser.parse_args()
    
    trainer(args)

if __name__ == "__main__":
    main()
