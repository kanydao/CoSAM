import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from tqdm import tqdm
import random
import torch.nn.functional as F
from scipy.ndimage import zoom
import json

my_root = '/mnt/889cdd89-1094-48ae-b221-146ffe543605/gwd/dataset/RecT500'


class uniform_dataset(Dataset):
    """
    ***请注意***
    3D影像的格式: D x H x W
    候选框的格式: z, x, y, d, h, w
    """

    def __init__(self, root_dir, roi_size=(128, 128, 128), mode="train"):
        super(uniform_dataset, self).__init__()
        assert mode in ["train", "test"]
        assert ((np.array(roi_size) % 16) == 0).all(), 'ROI size must be divisible by 16.'
        self.roi_size = roi_size
        self.scale = 2
        self.training = mode == "train"
        self.root_dir = root_dir
        self._build_dataset_from_csv()
        self.target_spacing = (1.25, 0.7, 0.7)

    def _build_box_and_transform(self, bbox, margin=1):
        """
        z1, y1, x1, z2, y2, x2 --> cz, cy, cx, d, h, w
        """
        bbox = np.array(bbox, dtype=np.int32)
        z1, y1, x1, z2, y2, x2 = bbox
        d, h, w = z2 - z1, y2 - y1, x2 - x1
        cz, cy, cx = (z1 + z2) / 2, (y1 + y2) / 2, (x1 + x2) / 2
        bbox = np.array([cz, cy, cx, d, h, w], dtype=float)
        bbox[3:] += margin * 2  # add margin
        return bbox

    def _build_dataset_from_csv(self):
        lymphlist = open(os.path.join(self.root_dir, 'PRLN', 'prln_target.csv'), 'r').readlines()
        with open(os.path.join(self.root_dir, 'PRLN', 'annotations.json')) as f:
            case_dict = json.load(f)

        if self.training:
            caselist = case_dict['train_normal'] + case_dict['train_both']
        else:
            caselist = case_dict['test_normal'] + case_dict['test_both']

        lymph_info_dict = {}
        for case in lymphlist:
            _case = case.strip().replace(' ', '').split(',')[:3]
            casename, label_id, vol = _case
            if casename in lymph_info_dict:
                lymph_info_dict[casename].append(case)
            else:
                lymph_info_dict[casename] = [case]

        box_dict = {}
        case_set = set()
        for case in caselist:
            target_list = lymph_info_dict.get(case)
            if target_list is None:
                continue

            for target in target_list:
                target = target.strip().replace(' ', '').split(',')
                casename, label_id, vol, bbox = target[0], target[1], target[2], target[3:]
                bbox = self._build_box_and_transform(bbox)[np.newaxis, ...]
                if box_dict.get(casename) is None:
                    box_dict[casename] = bbox
                else:
                    box_dict[casename] = np.vstack((box_dict[casename], bbox))
                case_set.add(casename)

        self.caselist = list(case_set)
        self.box_dict = box_dict

    def __len__(self):
        return len(self.caselist)

    def __getitem__(self, idx):
        if self.training:
            pid = casename = self.caselist[idx]
        else:
            pid = casename = self.caselist[idx]
        bboxes = self.box_dict[casename]
        npz = np.load(os.path.join(self.root_dir, 'npz', casename + '.npz'))
        img_np = npz['img']
        spacing = npz['spacing']
        npz = np.load(os.path.join(self.root_dir, 'clean_labels', casename + '.npz'))
        scale = np.array(spacing) / np.array(self.target_spacing) * self.scale
        # scale = self.scale
        bboxes *= np.concatenate((scale, scale), axis=0)
        img_np = zoom(img_np, scale, order=1)

        # choose a target as origin
        if self.training:
            orig = random.choice(bboxes)
        else:
            orig = bboxes[0]    # must be certain while testing

        roi = np.array(self.roi_size)
        fulsize = np.maximum(img_np.shape, roi)
        pos0 = orig[:3]
        vol = orig[3:]  # vol can be decimal
        pos1 = pos0 - roi / 2
        pos2 = pos0 + roi / 2
        shift_min = np.maximum(-pos1, pos0 + vol / 2 - pos2)
        shift_max = np.minimum(fulsize - pos2, pos0 - vol / 2 - pos1)
        shift_range = shift_max - shift_min
        assert (shift_range >= 0).all(), \
            ('Upper bound must be greater than lower bound!', orig, shift_min, shift_max)
        shift = np.zeros(3, dtype=np.int32)
        if self.training:
            # compatible with low version of numpy
            # shift = np.random.randint(shift_range + 1)   # +1 cause upper bound is exclusive
            for i in range(3):
                # +1 cause upper bound is exclusive
                shift[i] = np.random.randint(shift_range[i] + 1)
        else:
            # for eval, shift must be certain
            shift = np.maximum(-shift_min, 0)

        cor1 = (pos1 + shift_min + shift).astype('int32')
        cor2 = (pos2 + shift_min + shift).astype('int32')
        cor1 = np.maximum(cor1, 0)
        cor2 = np.minimum(cor2, img_np.shape)
        img_np = img_np[cor1[0]:cor2[0], cor1[1]:cor2[1], cor1[2]:cor2[2]]
        # clip & normalizae
        HU_MIN = -100
        HU_MAX = 100
        img_np = np.clip(img_np, HU_MIN, HU_MAX)
        img_np = (img_np - HU_MIN) / (HU_MAX - HU_MIN)
        # padding to roi size
        pad1 = (roi - img_np.shape) // 2
        pad2 = roi - img_np.shape - pad1
        padding = (pad1[2], pad2[2], pad1[1], pad2[1], pad1[0], pad2[0])
        img_tensor = F.pad(torch.tensor(img_np), padding, mode='constant', value=0)
        assert (np.array(img_tensor.shape) == roi).all(), 'Error occurs during padding.'
        # filter bboxes in roi
        bbox_area = bboxes[:, 3] * bboxes[:, 4] * bboxes[:, 5]
        if not (bbox_area > 0).all():
            print('Error occurs in case ' + casename)
            print(bboxes)
        bbox_c1 = bboxes[:, :3] - bboxes[:, 3:] / 2
        bbox_c2 = bboxes[:, :3] + bboxes[:, 3:] / 2
        intersect_c1 = np.maximum(cor1 - pad1, bbox_c1)
        intersect_c2 = np.minimum(cor2 + pad2, bbox_c2)
        intersect_size = np.maximum(intersect_c2 - intersect_c1, 0)
        intersect_area = intersect_size[:, 0] * intersect_size[:, 1] * intersect_size[:, 2]
        ratio = intersect_area / bbox_area
        INTERSECT_THRESH = 0.8
        bboxes = bboxes[ratio >= INTERSECT_THRESH]
        bboxes[:, :3] = bboxes[:, :3] - cor1 + pad1
        bboxes = torch.from_numpy(bboxes).clip(0, self.roi_size[0] - 1)
        img_tensor = img_tensor.unsqueeze(0).float()
        labels = torch.ones(len(bboxes), dtype=torch.long)
        return [img_tensor, bboxes, labels, pid]


def uniform_collate(batch):
    batch_size = len(batch)
    imgs = torch.stack([batch[b][0] for b in range(batch_size)], 0).float()
    bboxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]
    pids = [batch[b][3] for b in range(batch_size)]
    return [imgs, bboxes, labels, pids]
