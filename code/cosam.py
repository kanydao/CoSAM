"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union
from torchvision.models.detection.image_list import ImageList
from torch.nn import functional as F
from lens import SAM
import copy
from skimage import measure
from util import misc as utils
import numpy as np
from utils import non_max_suppression
from refiner import joint_classifier


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, channel_attn, rpn, roi_heads, transform, cosam=False):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.channel_attn = channel_attn
        self._has_warned = False
        self.wsize = roi_heads.wsize
        self.cosam = cosam
        '''joint learning modules begin'''
        if self.cosam:
            # parse args
            from argparse import Namespace
            args = Namespace()
            args.sam_num_classes = 2
            args.use_gt_box = True
            args.use_gt_pts = False
            args.use_psd_box = False
            args.use_psd_pts = False
            args.use_psd_mask = False
            args.img_size = 224
            # load sam model
            self.sam_model = SAM(args)
            print('Load SAM checkpoint')
            usam_ckpt = 'weight/usam.pth'
            with open(usam_ckpt, 'rb') as f:
                ckpt = torch.load(f, map_location='cpu')
                self.sam_model.load_state_dict(ckpt['model'], strict=False)
            self.sam_model.pixel_mean = transform.image_mean
            self.sam_model.pixel_std = transform.image_std
            self.sam_head = sam_head(self.sam_model, self.roi_heads.batch_size_per_image)
            # load frcnn model
            print('Load FRCNN checkpoint')
            frcnn_ckpt = 'weight/frcnn.pth'
            with open(frcnn_ckpt, 'rb') as f:
                ckpt = torch.load(f, map_location='cpu')
                self.load_state_dict(ckpt['model'], strict=False)
    
    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        # print(list([tuple(images.tensors[-1].shape[-2:])]))
        curr_idx = images.tensors.shape[0] // 2
        current_image = ImageList(images.tensors[curr_idx].unsqueeze(0), list([tuple(images.tensors[-1].shape[-2:])]))

        if targets is None:
            current_target = None
        else:
            current_target = list([targets[0]])

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        context_features = self.backbone(images.tensors)
        keys = list(context_features.keys())
        vals = list(context_features.values())
        alpha = 0.5
        wsize = 5
        hw = wsize // 2
        avgpool_features = []
        for k, v in zip(keys, vals):
            curr_feat = v[curr_idx].unsqueeze(0)
            avg_context_feat = torch.mean(v[curr_idx - hw: curr_idx + hw + 1], dim=0, keepdim=True)
            avg_feat = alpha * avg_context_feat + (1 - alpha) * curr_feat
            avgpool_features.append((k, avg_feat))
        avgpool_features = OrderedDict(avgpool_features)

        proposals, proposal_losses = self.rpn(current_image, avgpool_features, current_target)
        detections, detector_losses = self.roi_heads(context_features, proposals, images.image_sizes, current_target)

        if self.cosam:
            refiner_loss, detections = self.sam_head(detections, proposals, targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        if self.cosam:
            losses.update(refiner_loss)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)



class sam_head(nn.Module):
    def __init__(self, sam_model, batch_size_per_image) -> None:
        super().__init__()
        self.sam_model = sam_model
        self.batch_size_per_image = batch_size_per_image
        self.joint_classifier = joint_classifier()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, detections, proposals, targets):
        pred_boxes = detections[0]['boxes'] * 224 / 800
        sequence_features = detections[0]['features'].squeeze()
        positive_inds = detections[0]['positive_inds']
        selected_inds = detections[0]['selected_inds']
        # logits = detections[0]['logits'].clone().detach()
        device = pred_boxes.device
        num_proposals = self.batch_size_per_image if self.training else len(proposals[0])
        labels = torch.zeros(num_proposals, device=device)
        labels[positive_inds] += 1
        gt_labels = torch.where(labels[selected_inds]  > 0, 1, 0).long()
        # use mini-batch for segmentation
        num_boxes = len(pred_boxes)
        mini_batch = 12
        pred_boxes = torch.chunk(pred_boxes, mini_batch, dim=0)
        gt_labels = torch.chunk(gt_labels, mini_batch, dim=0)
        batch_num = len(pred_boxes)
        masks = None
        sam_loss = None
        segment_features = None
        for b in range(batch_num):
            input_slice = targets[0]['slice'].unsqueeze(0).repeat(len(pred_boxes[b]), 1, 1, 1)
            _targets = []
            lbl = measure.label(targets[0]['label'].cpu().numpy())
            regs = measure.regionprops(lbl)
            for idx, box in enumerate(pred_boxes[b]):
                _target = copy.deepcopy(targets[0])
                _target['orig_boxes'] = box[[1, 0, 3, 2]].unsqueeze(0).int()
                _target['box_label'] = gt_labels[b][idx]
                if not gt_labels[b][idx]:
                    _target['mask'] = torch.zeros((224, 224), dtype=torch.long)
                else:
                    _box = box[[1, 0, 3, 2]].int().cpu()
                    ious = [utils.compute_iou(reg.bbox, _box) for reg in regs]
                    _ind_max = np.argmax(ious)
                    _reg = regs[_ind_max]
                    _mask = np.zeros((224, 224), dtype=int)
                    for x, y in _reg.coords:
                        _mask[x, y] = 1
                    _target['mask'] = torch.tensor(_mask).long()
                _targets.append(_target)
            targets = [{k: v.to(device) for k, v in t.items()} for t in _targets]
            if self.training:
                _masks, _sam_loss, _, _, _segment_features = self.sam_model(input_slice, targets)
                if sam_loss is None:
                    sam_loss = _sam_loss
                else:
                    sam_loss += _sam_loss
            else:
                _masks, _, _, _, _, _segment_features = self.sam_model(input_slice, targets)
                _masks = _masks.argmax(dim=1, keepdim=False)

            if masks is None:
                masks = _masks
            else:
                masks = torch.cat([masks, _masks])

            if segment_features is None:
                segment_features = _segment_features
            else:
                segment_features = torch.cat([segment_features, _segment_features])

        assert masks.shape[0] == num_boxes, 'masks shape: {}, num_boxes: {}'.format(masks.shape, num_boxes)
        cls_logits = self.joint_classifier(sequence_features, segment_features)

        refiner_loss = {}
        if self.training:
            gt_labels = torch.where(labels[selected_inds] > 0, 1, 0).long()
            refiner_loss = self.ce_loss(cls_logits, gt_labels)
            # refiner_loss = torch.tensor(0, device=device)
            refiner_loss = {'refiner_loss': refiner_loss,
                            'sam_loss': sam_loss}
        else:
            scores = F.softmax(cls_logits, dim=1)[:, 1]
            # scores = F.softmax(logits, dim=1)[:, 1]
            sorted_inds = torch.argsort(scores, descending=True)
            det = {}
            det['boxes'] = detections[0]['boxes'][sorted_inds]
            det['labels'] = detections[0]['labels'][sorted_inds]
            det['scores'] = scores[sorted_inds]
            masks = masks[sorted_inds]
            score_thresh = 0.7 * scores[0].cpu().item()
            # score_thresh = 0.5
            nms_boxes = det['boxes'].clone().detach().cpu().numpy()
            nms_scores = det['scores'].clone().detach().cpu().numpy()
            nms_kept_inds = torch.tensor(
                non_max_suppression(nms_boxes, nms_scores, iou_threshold=0.15, score_threshold=score_thresh),
                device=device, dtype=torch.long)
            _mask = torch.zeros((224, 224), device=device, dtype=torch.long)
            for idx in nms_kept_inds:
                _mask = torch.logical_or(_mask, masks[idx])

            det['mask'] = _mask
            det['boxes'] = det['boxes']
            det['scores'] = det['scores']
            detections = [det]
        
        return refiner_loss, detections

