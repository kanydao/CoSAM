import copy
import os
import random
from glob import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations import RandomBrightnessContrast, RandomGamma, CLAHE, Compose
from PIL import Image
from tqdm import tqdm
from skimage import measure
from scipy.ndimage import zoom
from util import misc as utils
from monai.transforms import (
    Compose,
    AddChanneld,
    RandRotate90d,
    RandAdjustContrastd,
    RandFlipd,
    Resized,
    ToTensord,
)

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class lymph_dataset(Dataset):
    def __init__(self, root_dir, image_size=(224, 224), roi_size=(112, 112), wsize=15, mode="train"):

        """
        root_dir :
              Path of data folder containing train and test images in separate folders.
        image_size:
              all images will be resized to this
        """
        super(lymph_dataset, self).__init__()

        assert mode in ["train", "test"]
        self.wsize = wsize
        self.roi_size = roi_size
        self.image_size = image_size
        self.training = mode == "train"

        # data_dir = os.path.join(root_dir, mode, '*.npz')
        # self.datalist = sorted(glob(data_dir))

        self.datalist = open(os.path.join(root_dir, f'caselist_{mode}.txt')).readlines()
        self.root_dir = os.path.join(root_dir, mode)

    def __len__(self):
        """return the length of the dataset"""
        return len(self.datalist)

    def __getitem__(self, idx):
        # filename = self.datalist[idx]
        filename = os.path.join(self.root_dir, self.datalist[idx].strip() + '.npz')
        npz = np.load(filename)
        imgs = npz['img']
        lbls = npz['label']
        _lbls = np.zeros_like(lbls)
        _lbls[lbls > 0] = 1
        del lbls
        lbls = _lbls
        hw = self.wsize // 2
        HW = 15 // 2
        # select region larger than 50
        lbl_slc = lbls[..., HW]
        lbl_map = measure.label(lbl_slc)
        regs = measure.regionprops(lbl_map)
        regs = [reg for reg in regs if reg.area >= 50]
        if self.training:
            reg = random.sample(regs, 1)[0]
        else:
            reg = regs[0]
        del regs

        # select a point as center in reg
        if self.training:
            cx, cy = random.sample(reg.coords.tolist(), 1)[0]
        else:
            cx, cy = np.array(reg.centroid).astype(int)

        h, w = self.roi_size
        H, W = lbl_slc.shape

        while True:
            if self.training:
                ds = 32
                rds1 = random.randint(-ds, ds)
                rds2 = random.randint(-ds, ds)
                cxx = cx + rds1
                cyy = cy + rds2
            else:
                cxx, cyy = cx, cy
            x1 = min(H - h, max(0, cxx - h // 2))
            x2 = max(h, min(H, cxx + h // 2))
            y1 = min(W - w, max(0, cyy - w // 2))
            y2 = max(w, min(W, cyy + w // 2))

            if not self.training:
                break

            xx1, yy1, xx2, yy2 = reg.bbox
            if xx1 > x1 and xx2 < x2 and yy1 > y1 and yy2 < y2:
                break

        imgs = imgs[x1:x2, y1:y2, :]
        lbls = lbls[x1:x2, y1:y2, :]

        H, W = self.image_size

        '''Monai'''
        sample = {'image': imgs, 'label': lbls}
        train_transform = Compose([
            AddChanneld(keys=['image', 'label']),
            Resized(
                keys=['image', 'label'],
                spatial_size=(H, W, 15),
                mode=['area', 'nearest']
            ),
            RandRotate90d(
                keys=['image', 'label'],
                prob=0.50,
                max_k=3,
            ),
            RandFlipd(
                keys=['image', 'label'],
                prob=0.50,
                spatial_axis=[0, 1]
            ),
            RandAdjustContrastd(
                keys='image',
                prob=0.50,
                gamma=1.5
            ),
            ToTensord(keys=['image', 'label']),
        ])
        test_transform = Compose([
            AddChanneld(keys=['image', 'label']),
            Resized(
                keys=['image', 'label'],
                spatial_size=(H, W, 15),
                mode=['area', 'nearest']
            ),
            ToTensord(keys=['image', 'label']),
        ])
        if self.training:
            sample = train_transform(sample)
        else:
            sample = test_transform(sample)
        sample['slice'] = sample['image'][..., HW].repeat(3, 1, 1)
        sample['image'] = sample['image'][..., HW - hw:HW + hw + 1].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        sample['label'] = sample['label'][..., HW].squeeze().long()
        lbl_map = measure.label(sample['label'])
        regs = measure.regionprops(lbl_map)
        regs = [reg for reg in regs
                if reg.bbox[0] > 0 and reg.bbox[2] < H and reg.bbox[1] > 0 and reg.bbox[3] < W]

        # generate bbox
        def set_box_margin(box, mg=16):
            x1, y1, x2, y2 = box
            x1 = max(0, x1 - mg)
            x2 = min(H, x2 + mg)
            y1 = max(0, y1 - mg)
            y2 = min(W, y2 + mg)
            return [x1, y1, x2, y2]

        def shift_box(box, dx=32):
            x1, y1, x2, y2 = box
            x1 = max(0, x1 + dx)
            x2 = min(H, x2 + dx)
            y1 = max(0, y1 + dx)
            y2 = min(W, y2 + dx)
            return [x1, y1, x2, y2]
        
        # for segmentation
        boxes = [set_box_margin(reg.bbox) for reg in regs]
        neg_ratio = 0.0
        background_thresh = 0.3
        segment = {'label': sample['label'], 'mask': sample['label']}
        _reg = random.sample(regs, 1)[0]
        _lbl = torch.zeros_like(sample['label'], dtype=torch.long)
        if self.training and random.random() < neg_ratio:
            # construct negative example
            while True:
                _box = shift_box(_reg.bbox, dx=random.randint(-64, 64))
                _x1, _y1, _x2, _y2 = _box
                
                if _x1 > 0 and _x2 < H and _y1 > 0 and _y2 < W:
                    max_iou = np.max([utils.compute_iou(_box, box) for box in boxes])
                    if max_iou < background_thresh:
                        break
            _lbl[_x1:_x2, _y1:_y2] = sample['label'][_x1:_x2, _y1:_y2].clone()
            segment['mask'] = _lbl
            segment['orig_boxes'] = torch.tensor(_box, dtype=torch.int).unsqueeze(0)
            segment['box_label'] = torch.tensor(0).float().unsqueeze(0)
        else:
            # construct positive example
            for x, y in _reg.coords:
                _lbl[x, y] = 1
            segment['mask'] = _lbl
            segment['orig_boxes'] = torch.tensor(
                set_box_margin(_reg.bbox, mg=random.randint(0, 20)), dtype=torch.int).unsqueeze(0)
            segment['box_label'] = torch.tensor(1).float().unsqueeze(0)
            
        segment['mask'] = torch.tensor(zoom(segment['mask'], (224 / H, 224 / W), order=0)).long()

        # for detection
        boxes = torch.tensor(boxes, dtype=torch.float32)
        inds = [1, 0, 3, 2]
        boxes = boxes[:, inds]
        del regs

        # calc area
        area = np.zeros(boxes.size(0))
        for i in range(boxes.size(0)):
            area[i] = (boxes[i][2] - boxes[i][0] + 1) * (boxes[i][3] - boxes[i][1] + 1)
        area = torch.tensor(area, dtype=torch.int64)
        # create labels
        labels = torch.ones(boxes.size(0), dtype=torch.int64)
        # image_id
        image_id = torch.tensor([idx])
        # iscrowd
        iscrowd = torch.zeros(boxes.size(0), dtype=torch.int64)
        # create target dictionary
        target = {"boxes": boxes,
                  "labels": labels,
                  "image_id": image_id,
                  "area": area,
                  "iscrowd": iscrowd}
        target.update(segment)
        if not self.training:
            npz = np.load(f'./det/{idx:03d}.npz')
            target['pred'] = torch.tensor(npz['pred'])
            target['score'] = torch.tensor(npz['score'])

        '''visualization'''
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 20))
        # plt.subplot(1, 2, 1)
        # plt.title('img')
        # plt.imshow(img_slc.numpy(), cmap='gray')
        # # for idx in range(boxes.size(0)):
        # #     _x1, _y1, _x2, _y2 = boxes[idx]
        # #     plt.plot([_x1, _x1, _x2, _x2, _x1], [_y1, _y2, _y2, _y1, _y1], 'r-')
        
        # plt.subplot(1, 2, 2)
        # plt.title('lbl')
        # plt.imshow(lbl_slc.numpy(), cmap='inferno', interpolation="nearest")
        # plt.show()

        return sample, target


if __name__ == '__main__':
    root_dir = '../../dataset/lymph'
    mode = 'test'
    dataset = lymph_dataset(root_dir=root_dir, mode=mode)
    dataloader = DataLoader(dataset, batch_size=1)

    for idx, data in tqdm(enumerate(dataloader)):
        pass