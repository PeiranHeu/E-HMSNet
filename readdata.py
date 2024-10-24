import numpy as np
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyData(Dataset):
    def __init__(self, root, mode='train'):
        assert mode in ['train', 'test'], f'{mode} not support.'
        self.mode = mode

        self.root = root
        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # 读取文件夹下的所有文件名
        with open(os.path.join(self.root, 'file_names.txt'), 'r') as f:
            self.infos = f.readlines()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_path = self.infos[index].strip()
        # image = Image.open(os.path.join(self.root, 'RGB', image_path + '_rgb.png'))
        # depth = Image.open(os.path.join(self.root, 'TIR', image_path + '_th.png')).convert('RGB')
        # label = Image.open(os.path.join(self.root, 'labels', image_path + '.png'))
        image = Image.open(os.path.join(self.root, 'RGB', image_path + '.jpg'))
        depth = Image.open(os.path.join(self.root, 'TIR', image_path + '.jpg')).convert('RGB')
        label = Image.open(os.path.join(self.root, 'labels', image_path + '.png'))

        sample = {
            'image': image,
            'depth': depth, # depth is TIR image.
            'label': label,
        }

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = self.processSUIMDataRFHW(sample['label'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        # sample['label_path'] = image_path.strip().split('/')[-1] + '.png'
        return sample

    def getRobotFishHumanReefWrecks(self, mask):
        # for categories: HD, RO, FV, WR, RI
        imw, imh = mask.shape[0], mask.shape[1]
        labels = np.zeros((imw, imh))
        for i in range(imw):
            for j in range(imh):
                if (mask[i, j, 0] == 0 and mask[i, j, 1] == 0 and mask[i, j, 2] == 0):
                    labels[i, j] = 0
                elif (mask[i, j, 0] == 0 and mask[i, j, 1] == 0 and mask[i, j, 2] == 1):
                    labels[i, j] = 1
                elif (mask[i, j, 0] == 0 and mask[i, j, 1] == 1 and mask[i, j, 2] == 0):
                    labels[i, j] = 2
                elif (mask[i, j, 0] == 0 and mask[i, j, 1] == 1 and mask[i, j, 2] == 1):
                    labels[i, j] = 3
                elif (mask[i, j, 0] == 1 and mask[i, j, 1] == 0 and mask[i, j, 2] == 0):
                    labels[i, j] = 4
                elif (mask[i, j, 0] == 1 and mask[i, j, 1] == 0 and mask[i, j, 2] == 1):
                    labels[i, j] = 5
                elif (mask[i, j, 0] == 1 and mask[i, j, 1] == 1 and mask[i, j, 2] == 0):
                    labels[i, j] = 6
                elif (mask[i, j, 0] == 1 and mask[i, j, 1] == 1 and mask[i, j, 2] == 1):
                    labels[i, j] = 7
        return labels

    def processSUIMDataRFHW(self, mask):
        # scaling image data and masks
        mask = np.array(mask) / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        m = self.getRobotFishHumanReefWrecks(mask)
        return m