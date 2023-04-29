from monai.data import Dataset
from monai.transforms import *
import numpy as np
import torch
from torch.nn.functional import interpolate


class MRIDataset(Dataset):

    def __init__(self, data, stack, transform):
        '''
        :param data: data list [{'img': path, 'label': path},{}]
        :param stack: stack num
        :param transform:
        '''
        self.data = data
        self.stack = stack
        self.transform = transform
        self.load = LoadImage()

        self._build_stack_img()

    def _build_stack_img(self):
        self.new_data = []
        for d in self.data:
            img = d['img']
            imgarr = self.load(img)  # imgarr[0]: array, imgarr[1]: information
            imgname = imgarr[1]['filename_or_obj']
            img_channel = np.moveaxis(imgarr[0], -1, 0)


            img_channel = torch.from_numpy(img_channel).unsqueeze(0).unsqueeze(0)
            img_resize = interpolate(
                input=img_channel,  # type: ignore
                size=(20, 128, 128),
            )
            img_resize = img_resize.squeeze()

            start_point = int((20 - self.stack) / 2)
            img_stack = img_resize[start_point: start_point + self.stack, ...]
            data_dict = {'img': img_stack, 'label': d['label'], 'name': imgname}
            self.new_data.append(data_dict)
        return self.new_data

    # get data operation
    def __getitem__(self, index):
        data = self.new_data[index]
        augments_data = self.transform(data)
        return augments_data

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.new_data)


if __name__ == '__main__':
    from glob import glob
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import monai

    imgs = glob('../data/*.gz')
    data = [{'img': i, 'label': l} for i, l in zip(imgs[:12], imgs[:12])]

    trans = Compose(
        [
            ScaleIntensityd(keys='img'),
            RandGaussianNoised(keys='img', prob=0.5, mean=0.0, std=0.1),
            RandFlipd(keys='img', prob=0.2, spatial_axis=0),
            RandFlipd(keys='img', prob=0.2, spatial_axis=1),
            Rand2DElasticd(keys='img', prob=1, spacing=(20, 20), magnitude_range=(1, 2)),
            EnsureTyped(keys=['img', 'label'])
        ]
    )
    dataset = MRIDataset(data, stack=10, transform=trans)
    check_loader = DataLoader(dataset, batch_size=10, num_workers=4, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    img = check_data['img'][0]
    plt.imshow(img[5], cmap='gray')