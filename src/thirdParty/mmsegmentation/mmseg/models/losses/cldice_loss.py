# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .utils import weight_reduce_loss


import torch
import torch.nn.functional as F

class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=40):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):

        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):

        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, img):
        
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):

        img1 = self.soft_open(img)
        skel = F.relu(img-img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):

        return self.soft_skel(img)
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1., exclude_background=False):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(pred, target):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((target * pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(target) + torch.sum(pred) + smooth)
    return (1. - coeff)

@MODELS.register_module()
class soft_dice_cldice(nn.Module):
    def __init__(self, 
                 iter_=3, 
                 alpha=0.5, 
                 smooth = 1., 
                 use_sigmoid=True,
                 exclude_background=False, 
                 loss_weight=1.0,
                 weight = None,
                 loss_name = 'loss_cldice',
                 line_index = 4):
        
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background
        self._loss_name = loss_name
        self.loss_weight = loss_weight
        self.line_index = line_index
        self.weight = weight
        self.use_sigmoid = use_sigmoid
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    
    def forward(self, 
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        # only compute cldice for the line structures base line_index
        # print("y_true", y_true)
        
        # print("pred", pred.shape)
        # print("target", target.shape)
        # print("pred ", pred)
        # print("target ", target)
        target_binary = (target == self.line_index).float().unsqueeze(1) # 形状变为 [B, 1, H, W]
        pred_logit = pred[:, self.line_index, :, :].unsqueeze(1) # 形状变为 [B, 1, H, W]
        pred_prob = torch.sigmoid(pred_logit)
        # y_true torch.Size([4, 5, 512, 512])
        # y_pred torch.Size([4, 512, 512])


        
        # print("pred_prob", pred_prob.shape)
        # print("target_binary", target_binary.shape)
        # print("pred_prob ", pred_prob)
        # print("target_binary ", target_binary)

        dice = soft_dice(pred_prob, target_binary)
        skel_pred = self.soft_skeletonize(pred_prob)
        skel_true = self.soft_skeletonize(pred_prob)
        tprec = (torch.sum(torch.multiply(skel_pred, target_binary))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, target_binary))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0 * (tprec * tsens)/(tprec+tsens)
        return self.loss_weight * (1.0-self.alpha)*dice+self.alpha*cl_dice