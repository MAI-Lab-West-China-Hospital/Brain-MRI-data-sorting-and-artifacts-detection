import logging
import os
import sys
import time
import csv
import argparse

import matplotlib.colors
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

import numpy as np
import random

import warnings
import torch
import torch.nn as nn
from monai.data import DataLoader, DistributedSampler, CacheDataset, Dataset, PersistentDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision

import monai
from monai.networks.nets import DenseNet121, resnet50, EfficientNetBN, SEResNet50, SENet154
from EfficientNet import MyEfficientNet
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.transforms import *
from monai.utils import set_determinism
from monai.data import CSVSaver
from ImageFolder import make_dataset
from ExtractSlicesd import ExtractSlicesd
from SwapDimd import SwapDimd

set_determinism(seed=2021)
warnings.filterwarnings('ignore')

num_slices = 3
gap = 2

def form_results(lr, epochs, num_slices, model):
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    date = time.strftime('%Y%m%d', time.localtime(time.time()))
    results_path = './results/'.format(gap) + model
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    folder_name = "/{0}_num_slices{1}_lr{2}_epochs{3}_pretrainTrue".format(date, num_slices, lr, epochs)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


def train(train_loader, val_loader, pretrained, architecture: str, classes, epochs):
    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # chose model
    # assert (architecture in ['resnet50', 'EfficientNetBN', 'EfficientNet', 'densenet121', 'SENet154', 'SEResNet50'])
    if architecture == 'resnet50':
        model = resnet50(spatial_dims=2, n_input_channels=num_slices, num_classes=len(classes),
                         pretrained=pretrained).to(device)

    elif architecture == 'densenet121':
        model = DenseNet121(spatial_dims=2, in_channels=num_slices, out_channels=len(classes),
                            pretrained=pretrained).to(device)

    elif architecture == 'EfficientNetBN':
        model = EfficientNetBN("efficientnet-b0", spatial_dims=2, in_channels=num_slices,
                               num_classes=len(classes), pretrained=pretrained).to(device)

    elif architecture == 'EfficientNet':
        model = MyEfficientNet("efficientnet-b0", spatial_dims=2, in_channels=num_slices,
                               num_classes=len(classes), pretrained=pretrained, dropout_rate=0.2).to(device)

    elif architecture == 'SEResNet50':
        model = SEResNet50(spatial_dims=2, in_channels=num_slices, num_classes=len(classes),
                           pretrained=pretrained).to(device)

    elif architecture == 'SENet154':
        model = SENet154(spatial_dims=2, in_channels=num_slices, num_classes=len(classes),
                         pretrained=pretrained).to(device)

    elif architecture == 'vgg16':
        # /home/sun/anaconda3/envs/monai/lib/python3.8/site-packages/torchvision/models/vgg.py
        model = torchvision.models.vgg16(pretrained=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, len(classes))
        model.to(device)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    auc_metric = ROCAUCMetric()

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    
    train_writer = SummaryWriter(os.path.join(tensorboard_path, 'train'))
    val_writer = SummaryWriter(os.path.join(tensorboard_path, 'test'))

    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            if (step + 1) % 10 == 0:
                print(f"{step + 1}/{epoch_len}, train_loss: {loss.item():.4f}")

        epoch_loss /= step
        train_writer.add_scalar("Epoch Loss", epoch_loss, epoch + 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")


        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                val_loss = loss_function(y_pred, y)
                val_writer.add_scalar("Epoch Loss", val_loss.item(), epoch + 1)
                # calculate accuracy
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)

                y_onehot = [post_label(i).to(device) for i in decollate_batch(y)]
                y_pred_act = [post_pred(i).to(device) for i in decollate_batch(y_pred)]  # one-hot format                

                auc_metric(y_pred_act, y_onehot) # use one-hot format for computing auc
                auc_result = auc_metric.aggregate()
               
                auc_metric.reset()
                del y_pred_act, y_onehot
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), saved_model_path + "/best_model.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )
                val_writer.add_scalar("val_accuracy", acc_metric, epoch + 1)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    train_writer.close()
    val_writer.close()

if __name__ == "__main__":
    import warnings
    import pandas as pd
    from sklearn.metrics import classification_report
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--cuda', default='0', type=str, help='gpu id')
    parser.add_argument('--dataroot', default='../MRIseq_classification/dataset')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--epoch', default=300, type=int, help='entire epoch number')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--model', default='EfficientNet', help='choose model')
    # architecture can be ['densenet121', 'SENet154', 'SEResNet50', 'vgg16']
    parser.add_argument('--pretrained', default=True, type=bool)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    extensions = ('.jpg', '.jpeg', '.png', '.tif', '.gz')

    # get address
    tensorboard_path, saved_model_path, log_path = form_results(lr=args.lr,
                                                                epochs=args.epoch,
                                                                num_slices=num_slices,
                                                                model=args.model)

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # get data
    img_label_dict, imgs, labels, classes, class_to_idx = make_dataset(args.dataroot, extensions=extensions)
    print(f'There are {len(classes)} classes, specifically：')
    print(class_to_idx)
    # {'FlairCor': 0, 'FlairTra': 1, 'MRA': 2, 'T1Cor': 3, 'T1Sag': 4, 'T1Tra': 5, 'T2Cor': 6, 'T2Sag': 7, 'T2Tra': 8}

    # save classes_to_idx into files
    csv_path = os.path.join(log_path, 'classes.csv')
    with open(csv_path, 'a') as file:
        w = csv.writer(file)
        for key, val in class_to_idx.items():
            # write every key and value to file
            w.writerow([key, val])

    # save args
    argsDict = args.__dict__
    argsPath = os.path.join(log_path, 'args.txt')
    with open(argsPath, 'w') as f:
        f.writelines('------------------ start --------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ':' + str(value) + '\n')
        f.writelines('------------------ end --------------' + '\n')

    # split 6:2:2
    X_train_val, X_test, y_train_val, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=42, stratify=labels)
                                                           
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=labels)
    
    # count the number of each class
    trainset = set(y_train)
    train_count = {}
    for cls in trainset:
        train_count.update({cls: [y_train.count(cls), labels.count(cls)]})
    print('number of train data/all data')
    print(train_count)

    valset = set(y_val)
    val_count = {}
    for cls in valset:
        val_count.update({cls: [y_val.count(cls), labels.count(cls)]})
    print('number of val data/all data')
    print(val_count)
    
    # build train files and val files
    train_files = []
    val_files = []
    for i, j in zip(X_train, y_train):
        train_files.append({'img': i, 'label': j})

    for i, j in zip(X_val, y_val):
        val_files.append({'img': i, 'label': j})

    # Define transforms for image
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
    
    post_pred = Compose([EnsureType(), Activations(softmax=True)])  
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=len(classes))])
    
    # Define dataset, data loader
    check_ds = MRIDataset(data=train_data[:10], stack=args.stack, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=10, num_workers=4, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print('check image shape and label: ')
    print(check_data["img"].shape, check_data["label"])
    
    # create a training data loader
    train_ds = PersistentDataset(train_files, transform=transforms, cache_dir='./cache1')
    train_loader = DataLoader(train_ds,batch_size=args.batch, num_workers = 8, pin_memory=torch.cuda.is_available())    
    
    # create a validation data loader
    val_ds = PersistentDataset(val_files, transform=transforms, cache_dir='./cache1')
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers = 8, pin_memory=torch.cuda.is_available())  

    train(train_loader, val_loader, args.pretrained, args.model, classes, args.epoch)
