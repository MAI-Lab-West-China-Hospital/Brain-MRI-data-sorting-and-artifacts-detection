import logging
import os
import sys
import time
import csv
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn as nn
from monai.data import DataLoader, DistributedSampler, CacheDataset, Dataset, PersistentDataset
import monai
from monai.networks.nets import DenseNet121, resnet50, EfficientNetBN, SEResNet50, SENet154
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.transforms import *
from monai.utils import set_determinism
from monai.data import CSVSaver
from ImageFolder import make_dataset
from EfficientNet import MyEfficientNet
from ExtractSlicesd import ExtractSlicesd
from SwapDimd import SwapDimd
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

set_determinism(seed=2021)
num_slices = 3
gap = 2

parser = argparse.ArgumentParser(description='evaluated parameters')
parser.add_argument('--cuda', default='0', type=str, help='gpu id')
parser.add_argument('--dataroot', default='../MRIseq_classification/dataset')
parser.add_argument('--classes', default='model/log/classes.csv', type=str)
parser.add_argument('--batch', default=256, type=int, help='batch size')
parser.add_argument('--model', default='EfficientNet', help='choose model') # architecture can be ['resnet50', 'EfficientNetBN', 'densenet121', 'SENet154', 'SEResNet50']
parser.add_argument('--save_path', default='./pretrained_model')
args = parser.parse_args("")

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
extensions = ('.jpg', '.jpeg', '.png', '.tif', '.gz')

# get all data
img_label_dict, imgs, labels, classes, class_to_idx = make_dataset(args.dataroot, extensions=extensions)

X_train_val, X_test, y_train_val, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=42,   
                                                          stratify=labels)
# count the number of each class
testset = set(y_test)
test_count = {}
for cls in testset:
    test_count.update({cls: [y_test.count(cls), labels.count(cls)]})
print('number of test data/all data')
print(test_count)
print(f'number of total test data: {len(y_test)}')

test_files = []
for i, j in zip(X_test, y_test):
    test_files.append({'img': i, 'label': j})

name=[]
for file_path in X_test:
    name.append(file_path) 

# transform and data loader
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
test_ds = Dataset(test_files, transform=transforms, random_state=42, stratify=labels)
test_loader = DataLoader(test_ds, batch_size=args.batch, num_workers=8, pin_memory=torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
architecture = args.model
# assert (architecture in ['resnet50', 'EfficientNetBN', 'densenet121', 'SENet154', 'SEResNet50'])
if architecture == 'resnet50':
    model = resnet50(spatial_dims=2, n_input_channels=num_slices, num_classes=len(classes),
                     pretrained=False).to(device)

elif architecture == 'densenet121':
    model = DenseNet121(spatial_dims=2, in_channels=num_slices, out_channels=len(classes),
                        pretrained=False).to(device)

elif architecture == 'EfficientNetBN':
    model = EfficientNetBN("efficientnet-b0", spatial_dims=2, in_channels=num_slices,
                           num_classes=len(classes), pretrained=False).to(device)
elif architecture == 'EfficientNet':
    model = MyEfficientNet("efficientnet-b0", spatial_dims=2, in_channels=num_slices,
                               num_classes=len(classes), pretrained=False, dropout_rate=0.2).to(device)
elif architecture == 'SEResNet50':
    model = SEResNet50(spatial_dims=2, in_channels=num_slices, num_classes=len(classes),
                       pretrained=False).to(device)

elif architecture == 'SENet154':
    model = SENet154(spatial_dims=2, in_channels=num_slices, num_classes=len(classes),
                     pretrained=False).to(device)

elif architecture == 'vgg16':
    model = torchvision.models.vgg16(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, len(classes))
    model.to(device)

model_path = os.path.join(args.save_path, 'best_model.pth')
model.load_state_dict(torch.load(model_path))
model.eval()

output_path = os.path.join(args.save_path, 'log')

predict = []
truth = []

with torch.no_grad():
    y_prob = torch.tensor([], dtype=torch.float32, device=device)
    y_true = torch.tensor([], dtype=torch.long, device=device)
    num_correct = 0.0
    metric_count = 0
    for test_data in test_loader:
        test_images, test_labels = test_data["img"].to(device), test_data["label"].to(device)
        test_outputs = model(test_images).argmax(dim=1).detach().cpu().numpy().tolist()
        true = test_data["label"].numpy().tolist()
        
        y_prob = torch.cat([y_prob, model(test_images)], dim=0)  # probability
        y_true = torch.cat([y_true, test_labels], dim=0)
        
        predict.extend(test_outputs)
        truth.extend(true)
        
# save predict with name
df = pd.DataFrame()
df['name'] = name
df['pred'] = predict
df['truth'] = truth
df.to_csv(os.path.join(output_path, 'predict.csv'), index=False)
df.head()

fn = [] 
fp = []   
ft = [] 
for i in range(len(predict)):
    if predict[i] != truth[i]:
        fn.append(name[i])
        fp.append(predict[i])
        ft.append(truth[i])
        
df_false = pd.DataFrame()
df_false['name'] = fn
df_false['pred'] = fp
df_false['truth'] = ft
df_false.to_csv(os.path.join(output_path, 'predict_false.csv'), index=False)
df_false

y_pred = y_prob.argmax(dim=1).detach().cpu().numpy()  # get class
y_nohot = y_true.detach().cpu().numpy()

# classes= ['FLAIR_cor','FLAIR_tra','MRA_head','Other','T1_tra','T1c_cor','T1c_sag','T1c_tra','T2_sag','T2_tra','artifact_Aliasing','artifact_Metal','artifact_Motion']
classes= ['FLAIR_cor','FLAIR_tra','MRA_head','Metal_FLAIR_tra','Metal_T1_sag','Metal_T1_tra','Metal_T1c_sag','Metal_T1c_tra','Metal_T2_tra','Other','T1_tra','T1c_cor','T1c_sag','T1c_tra','T2_sag','T2_tra','artifact_Aliasing','artifact_Motion']

# get orientation precision
classes_orient = ['tra', 'cor', 'sag', 'none']

# get orientation y_pred and truth
y_orient = np.empty(4419,)
y_orient[np.where(y_nohot==0)]=1
y_orient[np.where(y_nohot==11)]=1
y_orient[np.where(y_nohot==1)]=0
y_orient[np.where(y_nohot==3)]=0
y_orient[np.where(y_nohot==5)]=0
y_orient[np.where(y_nohot==7)]=0
y_orient[np.where(y_nohot==8)]=0
y_orient[np.where(y_nohot==10)]=0
y_orient[np.where(y_nohot==13)]=0
y_orient[np.where(y_nohot==15)]=0
y_orient[np.where(y_nohot==4)]=2
y_orient[np.where(y_nohot==6)]=2
y_orient[np.where(y_nohot==12)]=2
y_orient[np.where(y_nohot==14)]=2
y_orient[np.where(y_nohot==2)]=3
y_orient[np.where(y_nohot==9)]=3
y_orient[np.where(y_nohot==16)]=3
y_orient[np.where(y_nohot==17)]=3

y_pred_orient = np.empty(4419,)
y_pred_orient[np.where(y_pred==0)]=1
y_pred_orient[np.where(y_pred==11)]=1
y_pred_orient[np.where(y_pred==1)]=0
y_pred_orient[np.where(y_pred==3)]=0
y_pred_orient[np.where(y_pred==5)]=0
y_pred_orient[np.where(y_pred==7)]=0
y_pred_orient[np.where(y_pred==8)]=0
y_pred_orient[np.where(y_pred==10)]=0
y_pred_orient[np.where(y_pred==13)]=0
y_pred_orient[np.where(y_pred==15)]=0
y_pred_orient[np.where(y_pred==4)]=2
y_pred_orient[np.where(y_pred==6)]=2
y_pred_orient[np.where(y_pred==12)]=2
y_pred_orient[np.where(y_pred==14)]=2
y_pred_orient[np.where(y_pred==2)]=3
y_pred_orient[np.where(y_pred==9)]=3
y_pred_orient[np.where(y_pred==16)]=3
y_pred_orient[np.where(y_pred==17)]=3

cf_matrix = confusion_matrix(y_nohot,y_pred)
fig,ax = plot_confusion_matrix(conf_mat=cf_matrix,
                               colorbar=False,
                               show_absolute=True,
                               show_normed=False,
                               class_names=classes,
                               )

def calculate_specificity_multiclass(confusion_matrix,class_index):
    true_negatives=np.sum(confusion_matrix) - np.sum(confusion_matrix[class_index,:])-np.sum(confusion_matrix[:,class_index])
    false_positives=np.sum(confusion_matrix[:,class_index])-confusion_matrix[class_index,class_index]
    total_negatives=true_negatives/(true_negatives + false_positives)
    specificity=true_negatives/(true_negatives+false_positives)
    return specificity

specificities = []
for i in range(len(classes)):
    specificity=calculate_specificity_multiclass(cf_matrix,i)
    specificities.append(specificity)

for i, class_name in enumerate(classes):
    print(f'Specificity for {class_name}:{specificities[i]:.4f}')

# cal precision, recall, f1-score
metrics = classification_report(y_nohot, y_pred, target_names=classes, digits=3)
print(metrics)

# get orientation metric
cf_matrix = confusion_matrix(y_orient, y_pred_orient.round())
fig, ax = plot_confusion_matrix(conf_mat=cf_matrix,
                                colorbar=False,
                                show_absolute=True,
                                show_normed=False,
                                class_names=classes_orient,
                                )

metrics = classification_report(y_orient, y_pred_orient.round(), target_names=classes_orient, digits=3)
print(metrics)

post_pred = Compose([EnsureType(), Activations(softmax=True)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=len(classes))])

y_onehot = [post_label(i).cpu().numpy().tolist() for i in decollate_batch(y_true)]  # array
y_score = [post_pred(i).cpu().numpy().tolist() for i in decollate_batch(y_prob)]  # array

# list to array
y_onehot = np.array(y_onehot)
y_score = np.array(y_score)

# roc, auc
fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = len(classes)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
print(roc_auc)

# plot auc curve
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#1a55FF', '#9467bd', '#d62728', '#2ca02c', '#ff7f0e', '#8c564b', '#e377c2', '#bcbd22']
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i],tpr[i],lw=1,label='ROC curve of class {0} (AUC={1:0.2f})'.format(i, roc_auc[i]))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend()












