import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2

from monai.data import DataLoader, DistributedSampler, CacheDataset, Dataset, PersistentDataset
from monai.visualize import CAM, GradCAM, OcclusionSensitivity
from monai.transforms import *

from ImageFolder import make_dataset
from EfficientNet import MyEfficientNet
from ExtractSlicesd import ExtractSlicesd
from SwapDimd import SwapDimd
from EfficientNet import MyEfficientNet

import SimpleITK as sitk


num_slices = 3
gap = 2

# load model
device = torch.device("cpu")

model = MyEfficientNet("efficientnet-b0", spatial_dims=2, in_channels=num_slices,
                               num_classes=18, pretrained=False, dropout_rate=0.2).to(device)
save_path = './pretrained_model/'
model_path = os.path.join(save_path, 'best_model.pth')
model.load_state_dict(torch.load(model_path))
model.eval()
print('load model')
print(model)

# get module name
for name, _ in model.named_modules():
    print(name)
    
    
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
imgs = glob('../data/*.gz')
data = []
for i in imgs:
    data.append({'img': i, 'label': 0})  
ds = Dataset(data=data, transform=transforms)
loader = DataLoader(ds, batch_size=1, num_workers=1)

# get cam
gradcam = GradCAM(nn_module=model, target_layers='backbone._bn1')

cams = []
images = []
for d in loader:
    input, labels = d["img"].to(device), d["label"].to(device)
    cam_ =gradcam(input)
    cam_ = cam_.numpy().squeeze()
    image = d["img"].numpy().squeeze()
    images.append(image)
    cams.append(cam_)

num = 0
def get_sitk_image(path, img, name): #name:*nii.gz
    img_path = os.path.join(path, name)
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = sitk.GetImageFromArray(img)
    return sitk.WriteImage(img, img_path)

path = '../data/cam_images'
im_1 = get_sitk_image(path, images[num], name='img.nii.gz')
im_2 = get_sitk_image(path, cams[num], name='cmap.nii.gz')

im_path_1 = os.path.join(path,'img.nii.gz')
im_path_2 = os.path.join(path, 'cmap.nii.gz')

image_sitk = sitk.ReadImage(im_path_1, sitk.sitkFloat32)
image = sitk.GetArrayFromImage(image_sitk)

attention_map_sitk = sitk.ReadImage(im_path_2)
attention_map = sitk.GetArrayFromImage(attention_map_sitk)

slice=1
color = plt.get_cmap('jet')
rjet = color.reversed()

fig, axes = plt.subplots(1, 3, figsize=(15,15))
axes[0].imshow(np.squeeze(images[num][slice, :, :]), cmap='gray',origin='lower')
axes[0].axis('off')
        
axes[1].imshow(np.squeeze(image[slice, :, :]), cmap='gray',alpha=1,origin='lower') 
axes[1].imshow(np.squeeze(attention_map),cmap=rjet, alpha=0.3,origin='lower')
axes[2].imshow(np.squeeze(cams[num]),cmap=rjet, origin='lower')
axes[1].axis('off')
plt.subplots_adjust(wspace=0.05, hspace=0.05)
