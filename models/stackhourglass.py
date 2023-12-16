from __future__ import print_function

import math
import random
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.special import erfinv
from torch.autograd import Variable
from torchvision.transforms.functional_tensor import _hsv2rgb, _rgb2hsv

from .submodule import *


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

def rgb_to_grayscale(img):
    r, g, b = img.unbind(dim=-3)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)
    return l_img

class PSMNet(nn.Module):
    def __init__(self, maxdisp, args):
        super(PSMNet, self).__init__()
        self.args = args

        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        self.feature_discriminator = feature_discriminator()
        self.feature_discriminator_res18 = ResNet(BasicBlock_res, [2, 2, 2, 2], num_classes=4)
        weight = torch.load(self.args.res18)
        weight['fc.weight'] = self.feature_discriminator_res18.state_dict()['fc.weight']
        weight['fc.bias'] = self.feature_discriminator_res18.state_dict()['fc.bias']
        self.feature_discriminator_res18.load_state_dict(weight)

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.toPIL = transforms.ToPILImage(mode="RGB")
        self.trans = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.B_min = nn.Parameter(torch.randn(1))
        self.C_min = nn.Parameter(torch.randn(1))
        self.S_min = nn.Parameter(torch.randn(1))
        self.H_min = nn.Parameter(torch.randn(1))
        self.B_max = nn.Parameter(torch.randn(1))
        self.C_max = nn.Parameter(torch.randn(1))
        self.S_max = nn.Parameter(torch.randn(1))
        self.H_max = nn.Parameter(torch.randn(1))

        self.B_min_p = nn.Parameter(torch.randn(args.num_patch))
        self.C_min_p = nn.Parameter(torch.randn(args.num_patch))
        self.S_min_p = nn.Parameter(torch.randn(args.num_patch))
        self.H_min_p = nn.Parameter(torch.randn(args.num_patch))
        self.B_max_p = nn.Parameter(torch.randn(args.num_patch))
        self.C_max_p = nn.Parameter(torch.randn(args.num_patch))
        self.S_max_p = nn.Parameter(torch.randn(args.num_patch))
        self.H_max_p = nn.Parameter(torch.randn(args.num_patch))

        self.alpha_gaosi_l = nn.Parameter(torch.randn(1,3,256,512))
        self.alpha_gaosi_r = nn.Parameter(torch.randn(1,3,256,512))

    def img_global_generation(self, img_in, B, C, S, H):
        img_in = img_in.permute(2,0,1)
        img = (img_in.unsqueeze(0) / 255).to(img_in.device)
        # Random order
        idx_list = torch.randperm(4)
        # sub-transformations
        for i in idx_list:
            if i == 0:
                zero_img = torch.zeros_like(img).to(img_in.device)
                img = (B * img + (1.0 - B) * zero_img).clamp(0, 1.0).to(img.dtype)  # Brightness
            elif i == 1:
                mean_img = torch.mean(rgb_to_grayscale(img).to(img.dtype), dim=(-3, -2, -1), keepdim=True).to(img_in.device)
                img = (C * img + (1.0 - C) * mean_img).clamp(0, 1.0).to(img.dtype) # Contrast
            elif i == 2:
                satu_img = rgb_to_grayscale(img).to(img.dtype).to(img_in.device)
                img = (S * img + (1.0 - S) * satu_img).clamp(0, 1.0).to(img.dtype)  # Saturation
            elif i == 3:
                img = _rgb2hsv(img)                                                 # Hue
                h, s, v = img.unbind(dim=-3)
                h = (h + H) % 1.0
                img = torch.stack((h, s, v), dim=-3)
                img = _hsv2rgb(img)
        # Normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_in.device).view(-1,1,1).unsqueeze(0)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_in.device).view(-1,1,1).unsqueeze(0)
        out_ = (img - mean) / std
        return out_

    def img_local_generation(self, img_org, img, para_list):
        # the learned parameter in local visual transformation 
        [B_min_p,C_min_p,S_min_p,H_min_p,B_max_p,C_max_p,S_max_p,H_max_p] = para_list

        img_trans_list = []
        img_trans_list_disc_loss = []
        # h_p: h of patch  w_p: w of patch
        # patch_idx_list = torch.randperm(self.args.num_patch)
        split_num = int(sqrt(self.args.num_patch))
        _, _, h, w = img.shape
        h_p, w_p = int(h // split_num), int(w // split_num)

        for j in range(img_org.shape[0]):
            img_patch_list = []
            img_patch_list_g = []
            img_patch_list_g_loss = []
            # acquire the img patches
            for m in range(split_num):
                for n in range(split_num):
                    img_patch_list.append(img_org[j][m*h_p:(m+1)*h_p,n*w_p:(n+1)*w_p,:])

            img_local = torch.zeros([3,h,w], dtype=torch.float32).to(img_org.device)
            img_local_loss = torch.zeros([3,h,w], dtype=torch.float32).to(img_org.device)
            for i in range(0, self.args.num_patch):
                # the local transformed image  
                B = torch.empty(1, device=img_org.device).uniform_(float(B_min_p[i]), float(B_max_p[i]))
                C = torch.empty(1, device=img_org.device).uniform_(float(C_min_p[i]), float(C_max_p[i]))
                S = torch.empty(1, device=img_org.device).uniform_(float(S_min_p[i]), float(S_max_p[i]))
                H = torch.empty(1, device=img_org.device).uniform_(float(H_min_p[i]), float(H_max_p[i]))
                img_patch_list_g.append(self.img_global_generation(img_patch_list[i], B, C, S, H))
                img_local[:, (i//split_num)*h_p:(i//split_num+1)*h_p, (i%split_num)*w_p:(i%split_num+1)*w_p] = img_local[:, (i//split_num)*h_p:(i//split_num+1)*h_p, (i%split_num)*w_p:(i%split_num+1)*w_p] + img_patch_list_g[i].squeeze(0)

                '''
                to optimize the learned parameters in {I}_min and {I}_max, we generate the transformed image 
                with the min value of {I} (Probability = 0.5) or max value of {I} (Probability = 0.5). 
                the generated image represents the hardest training sample which the global image transformation module can produce.
                It is used to Maxmize the Cross-Domain Visual Discrepancy between original image and the global transformed image with loss_disc.

                I ∈ [B, C, S, H]
                ''' 
                m = torch.randn(1)
                if m > 0:
                    img_patch_list_g_loss.append(self.img_global_generation(img_patch_list[i], B_min_p[i], C_min_p[i], S_min_p[i], H_min_p[i]))
                else:
                    img_patch_list_g_loss.append(self.img_global_generation(img_patch_list[i], B_max_p[i], C_max_p[i], S_max_p[i], H_max_p[i]))
                img_local_loss[:, (i//split_num)*h_p:(i//split_num+1)*h_p, (i%split_num)*w_p:(i%split_num+1)*w_p] = img_local_loss[:, (i//split_num)*h_p:(i//split_num+1)*h_p, (i%split_num)*w_p:(i%split_num+1)*w_p] + img_patch_list_g_loss[i].squeeze(0)
            
            img_trans_list.append(img_local.unsqueeze(0))
            img_trans_list_disc_loss.append(img_local_loss.unsqueeze(0))

        img_local_final = torch.cat(img_trans_list, dim=0)
        img_local_loss_final = torch.cat(img_trans_list_disc_loss, dim=0)

        return img_local_final, img_local_loss_final
    
    def global_visual_transformation(self, left_org, right_org, left):
        imL_trans_list = []
        imR_trans_list = []
        imL_trans_list_disc_loss = []
        imR_trans_list_disc_loss = []

        # the generation of the upper and lower bounds of the adjustable range respectively 
        # for the Brightness (B), Contrast (C), Saturation (S) and Hue (H)
        B_min = 1 - (torch.sigmoid(self.B_min) * self.args.mu + self.args.beta)
        C_min = 1 - (torch.sigmoid(self.C_min) * self.args.mu + self.args.beta)
        S_min = 1 - (torch.sigmoid(self.S_min) * self.args.mu + self.args.beta)
        H_min = - (torch.sigmoid(self.H_min) * self.args.mu + self.args.beta)
        B_max = 1 + (torch.sigmoid(self.B_max) * self.args.mu + self.args.beta)
        C_max = 1 + (torch.sigmoid(self.C_max) * self.args.mu + self.args.beta)
        S_max = 1 + (torch.sigmoid(self.S_max) * self.args.mu + self.args.beta)
        H_max = (torch.sigmoid(self.H_max) * self.args.mu + self.args.beta)

        for i in range(left.shape[0]):
            # the global transformed image for left image
            B = torch.empty(1, device=left.device).uniform_(float(B_min), float(B_max))
            C = torch.empty(1, device=left.device).uniform_(float(C_min), float(C_max))
            S = torch.empty(1, device=left.device).uniform_(float(S_min), float(S_max))
            H = torch.empty(1, device=left.device).uniform_(float(H_min), float(H_max))
            imL_g = self.img_global_generation(left_org[i], B, C, S, H)
            imL_trans_list.append(imL_g)

            # the global transformed image for right image
            B = torch.empty(1, device=left.device).uniform_(float(B_min), float(B_max))
            C = torch.empty(1, device=left.device).uniform_(float(C_min), float(C_max))
            S = torch.empty(1, device=left.device).uniform_(float(S_min), float(S_max))
            H = torch.empty(1, device=left.device).uniform_(float(H_min), float(H_max))
            imR_g = self.img_global_generation(right_org[i], B, C, S, H)
            imR_trans_list.append(imR_g)
            
            '''
            to optimize the learned parameters in {I}_min and {I}_max, we generate the transformed image 
            with the min value of {I} (Probability = 0.5) or max value of {I} (Probability = 0.5). 
            the generated image represents the hardest training sample which the global image transformation module can produce.
            It is used to Maxmize the Cross-Domain Visual Discrepancy between original image and the global transformed image with loss_disc.

            I ∈ [B, C, S, H]
            ''' 
            m = torch.randn(1)  # Control the Probability
            if m > 0:
                imL_g_loss = self.img_global_generation(left_org[i], B_min, C_min, S_min, H_min)
                imR_g_loss = self.img_global_generation(right_org[i], B_min, C_min, S_min, H_min)
            else:
                imL_g_loss = self.img_global_generation(left_org[i], B_max, C_max, S_max, H_max)
                imR_g_loss = self.img_global_generation(right_org[i], B_max, C_max, S_max, H_max)
            imL_trans_list_disc_loss.append(imL_g_loss)    
            imR_trans_list_disc_loss.append(imR_g_loss)

        left_global = torch.cat(imL_trans_list, dim=0)
        right_global = torch.cat(imR_trans_list, dim=0)
        left_global_loss = torch.cat(imL_trans_list_disc_loss, dim=0)
        right_global_loss = torch.cat(imR_trans_list_disc_loss, dim=0)

        return left_global, right_global, left_global_loss, right_global_loss
        
    def max_discrepency_loss_stage1(self, left, right, left_global_loss, right_global_loss, b):
        # the ground truth domain label 0: original 1: global 2: local 3: pixel
        label_o_gt = torch.tensor([0], device=left.device).expand(b).long()
        label_g_gt = torch.tensor([1], device=left.device).expand(b).long() 

        # the feature and predicted domain label generated from the domain discriminating network 
        label_og_pred_l, feat_og_disc_l = self.feature_discriminator_res18(torch.cat([left, left_global_loss]))
        label_og_pred_r, feat_og_disc_r = self.feature_discriminator_res18(torch.cat([right, right_global_loss]))

        label_o_pred_l, label_g_pred_l, label_o_pred_r, label_g_pred_r = label_og_pred_l[0:b], label_og_pred_l[b:], label_og_pred_r[0:b], label_og_pred_r[b:]
        feat_o_disc_l, feat_g_disc_l, feat_o_disc_r, feat_g_disc_r = feat_og_disc_l[0:b], feat_og_disc_l[b:], feat_og_disc_r[0:b], feat_og_disc_r[b:]

        # the cross-entropy loss for domain classification
        loss_CE = (nn.CrossEntropyLoss()(label_o_pred_l, label_o_gt) + nn.CrossEntropyLoss()(label_o_pred_r, label_o_gt)  
                 + nn.CrossEntropyLoss()(label_g_pred_l, label_g_gt) + nn.CrossEntropyLoss()(label_g_pred_r, label_g_gt)) / 4
        loss_CE = loss_CE.sum() / b

        # the discrepency loss for maximizing cross-domain visual discrepancy by minimizing the consine similarity
        loss_disc = (torch.mean(torch.cosine_similarity(feat_g_disc_l, feat_o_disc_l, dim=1)) + torch.mean(torch.cosine_similarity(feat_g_disc_r, feat_o_disc_r, dim=1))) / 2
        
        return loss_CE, loss_disc
    
    def local_visual_transformation(self, left_org, right_org, left, right):
        
        # the generation of the upper and lower bounds of the adjustable range respectively 
        # for the Brightness (B), Contrast (C), Saturation (S) and Hue (H)
        B_min_p = 1 - (torch.sigmoid(self.B_min_p) * self.args.mu + self.args.beta)
        C_min_p = 1 - (torch.sigmoid(self.C_min_p) * self.args.mu + self.args.beta)
        S_min_p = 1 - (torch.sigmoid(self.S_min_p) * self.args.mu + self.args.beta)
        H_min_p = - (torch.sigmoid(self.H_min_p) * self.args.mu + self.args.beta)
        B_max_p = 1 + (torch.sigmoid(self.B_max_p) * self.args.mu + self.args.beta)
        C_max_p = 1 + (torch.sigmoid(self.C_max_p) * self.args.mu + self.args.beta)
        S_max_p = 1 + (torch.sigmoid(self.S_max_p) * self.args.mu + self.args.beta)
        H_max_p = (torch.sigmoid(self.H_max_p) * self.args.mu + self.args.beta)

        para_list = [B_min_p,C_min_p,S_min_p,H_min_p,B_max_p,C_max_p,S_max_p,H_max_p]
        left_local, left_local_loss = self.img_local_generation(left_org, left, para_list)
        right_local, right_local_loss = self.img_local_generation(right_org, right, para_list)

        return left_local, right_local, left_local_loss, right_local_loss
        
    def max_discrepency_loss_stage2(self, left, right, left_global_loss, right_global_loss, left_local_loss, right_local_loss, b):
        # the ground truth domain label 0: original 1: global 2: local 3: pixel
        label_o_gt = torch.tensor([0], device=left.device).expand(b).long()
        label_g_gt = torch.tensor([1], device=left.device).expand(b).long() 
        label_l_gt = torch.tensor([2], device=left.device).expand(b).long() 

        # the feature and predicted domain label generated from the domain discriminating network 
        label_ogl_pred_l, feat_ogl_disc_l = self.feature_discriminator_res18(torch.cat([left, left_global_loss, left_local_loss]))
        label_ogl_pred_r, feat_ogl_disc_r = self.feature_discriminator_res18(torch.cat([right, right_global_loss, right_local_loss]))

        label_o_pred_l, label_g_pred_l, label_l_pred_l, label_o_pred_r, label_g_pred_r, label_l_pred_r \
            = label_ogl_pred_l[0:b], label_ogl_pred_l[b:2*b], label_ogl_pred_l[2*b:], label_ogl_pred_r[0:b], label_ogl_pred_r[b:2*b], label_ogl_pred_r[2*b:]
        feat_o_disc_l, feat_g_disc_l, feat_l_disc_l, feat_o_disc_r, feat_g_disc_r, feat_l_disc_r \
            = feat_ogl_disc_l[0:b], feat_ogl_disc_l[b:2*b], feat_ogl_disc_l[2*b:], feat_ogl_disc_r[0:b], feat_ogl_disc_r[b:2*b], feat_ogl_disc_r[2*b:]

        # the cross-entropy loss for domain classification
        loss_CE = (nn.CrossEntropyLoss()(label_o_pred_l, label_o_gt) + nn.CrossEntropyLoss()(label_o_pred_r, label_o_gt)  
                    + nn.CrossEntropyLoss()(label_g_pred_l, label_g_gt) + nn.CrossEntropyLoss()(label_g_pred_r, label_g_gt)
                    + nn.CrossEntropyLoss()(label_l_pred_l, label_l_gt) + nn.CrossEntropyLoss()(label_l_pred_r, label_l_gt)) / 6
        loss_CE = loss_CE.sum() / b

        # the discrepency loss for maximizing cross-domain visual discrepancy by minimizing the consine similarity
        loss_disc = (torch.mean(torch.cosine_similarity(feat_g_disc_l, feat_o_disc_l, dim=1)) + torch.mean(torch.cosine_similarity(feat_g_disc_r, feat_o_disc_r, dim=1))
                   + torch.mean(torch.cosine_similarity(feat_l_disc_l, feat_o_disc_l, dim=1)) + torch.mean(torch.cosine_similarity(feat_l_disc_r, feat_o_disc_r, dim=1))) / 4
        
        return loss_CE, loss_disc
    
    def pixel_visual_transformation(self, left, right):
        
        gaosi_l = torch.randn(1,3,256,512, device=left.device) * (torch.sigmoid(self.alpha_gaosi_l) * self.args.mu + self.args.beta) 
        gaosi_r = torch.randn(1,3,256,512, device=left.device) * (torch.sigmoid(self.alpha_gaosi_r) * self.args.mu + self.args.beta) 
        left_pixel, right_pixel = left + gaosi_l, right + gaosi_r

        return left_pixel, right_pixel
        
    def max_discrepency_loss_stage3(self, left, right, left_global_loss, right_global_loss, left_local_loss, right_local_loss, left_pixel, right_pixel, b):
        # the ground truth domain label 0: original 1: global 2: local 3: pixel
        label_o_gt = torch.tensor([0], device=left.device).expand(b).long()
        label_g_gt = torch.tensor([1], device=left.device).expand(b).long() 
        label_l_gt = torch.tensor([2], device=left.device).expand(b).long() 
        label_p_gt = torch.tensor([3], device=left.device).expand(b).long() 

        # the feature and predicted domain label generated from the domain discriminating network 
        label_oglp_pred_l, feat_oglp_disc_l = self.feature_discriminator_res18(torch.cat([left, left_global_loss, left_local_loss, left_pixel]))
        label_oglp_pred_r, feat_oglp_disc_r = self.feature_discriminator_res18(torch.cat([right, right_global_loss, right_local_loss, right_pixel]))

        label_o_pred_l, label_g_pred_l, label_l_pred_l, label_p_pred_l, label_o_pred_r, label_g_pred_r, label_l_pred_r, label_p_pred_r \
            = label_oglp_pred_l[0:b], label_oglp_pred_l[b:2*b], label_oglp_pred_l[2*b:3*b], label_oglp_pred_l[3*b:], label_oglp_pred_r[0:b], label_oglp_pred_r[b:2*b], label_oglp_pred_r[2*b:3*b], label_oglp_pred_r[3*b:]
        feat_o_disc_l, feat_g_disc_l, feat_l_disc_l, feat_p_disc_l, feat_o_disc_r, feat_g_disc_r, feat_l_disc_r, feat_p_disc_r \
            = feat_oglp_disc_l[0:b], feat_oglp_disc_l[b:2*b], feat_oglp_disc_l[2*b:3*b], feat_oglp_disc_l[3*b:], feat_oglp_disc_r[0:b], feat_oglp_disc_r[b:2*b], feat_oglp_disc_r[2*b:3*b], feat_oglp_disc_r[3*b:]

        # the cross-entropy loss for domain classification
        loss_CE = (nn.CrossEntropyLoss()(label_o_pred_l, label_o_gt) + nn.CrossEntropyLoss()(label_o_pred_r, label_o_gt)  
                    + nn.CrossEntropyLoss()(label_g_pred_l, label_g_gt) + nn.CrossEntropyLoss()(label_g_pred_r, label_g_gt)
                    + nn.CrossEntropyLoss()(label_l_pred_l, label_l_gt) + nn.CrossEntropyLoss()(label_l_pred_r, label_l_gt)
                    + nn.CrossEntropyLoss()(label_p_pred_l, label_p_gt) + nn.CrossEntropyLoss()(label_p_pred_r, label_p_gt)) / 8
        loss_CE = loss_CE.sum() / b

        # the discrepency loss for maximizing cross-domain visual discrepancy by minimizing the consine similarity
        loss_disc = (torch.mean(torch.cosine_similarity(feat_g_disc_l, feat_o_disc_l, dim=1)) + torch.mean(torch.cosine_similarity(feat_g_disc_r, feat_o_disc_r, dim=1))
                   + torch.mean(torch.cosine_similarity(feat_l_disc_l, feat_o_disc_l, dim=1)) + torch.mean(torch.cosine_similarity(feat_l_disc_r, feat_o_disc_r, dim=1))
                   + torch.mean(torch.cosine_similarity(feat_p_disc_l, feat_o_disc_l, dim=1)) + torch.mean(torch.cosine_similarity(feat_p_disc_r, feat_o_disc_r, dim=1))) / 6
        
        return loss_CE, loss_disc

    def cost_volume_and_disparity_regression(self, refimg_fea, targetimg_fea, left):
        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp//4):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            distribute1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(distribute1)

            cost2 = torch.squeeze(cost2,1)
            distribute2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(distribute2)

        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        distribute3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(distribute3,dim=1)
        #For your information: This formulation 'softmax(c)' learned "similarity" 
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return [pred1, pred2, pred3]
        else:
            return [pred3]


    def forward(self, left, right, left_org, right_org, epoch, training=False):
        if training:

            stage_epoch = self.args.num_epoch // 3  # the number of epoches in each stage
            b,_,h,w = left.shape
            
            if epoch < stage_epoch:

                left_global, right_global, left_global_loss, right_global_loss = self.global_visual_transformation(left_org, right_org, left)

                refimg_fea_org_global     = self.feature_extraction(torch.cat([left, left_global], dim=0))
                targetimg_fea_org_global  = self.feature_extraction(torch.cat([right, right_global], dim=0))
                refimg_fea_org, targetimg_fea_org = refimg_fea_org_global[0:b], targetimg_fea_org_global[0:b]
                refimg_fea_global, targetimg_fea_global = refimg_fea_org_global[b:], targetimg_fea_org_global[b:]

                # the caculation of distance loss by minimizing cross-domain feature inconsistency between original feature and transformed feature
                loss_dist = torch.mean((refimg_fea_global - refimg_fea_org).pow(2)) + torch.mean((targetimg_fea_global - targetimg_fea_org).pow(2))
                # the caculation of cross-entropy loss and discrepency loss
                loss_CE, loss_disc = self.max_discrepency_loss_stage1(left, right, left_global_loss, right_global_loss, b)

                loss_hvt_stage1 = self.args.lambda_1 * loss_dist + self.args.lambda_2 * loss_disc + self.args.lambda_3 * loss_CE

                disp_ests_org = []
                disp_ests_global = []
                disp_ests_org_global = self.cost_volume_and_disparity_regression(refimg_fea_org_global, targetimg_fea_org_global, left)
                for disp_ests_tmp in disp_ests_org_global:
                    disp_ests_org.append(disp_ests_tmp[0:b])
                    disp_ests_global.append(disp_ests_tmp[b:])

                return [disp_ests_org, disp_ests_global], loss_hvt_stage1

            if epoch >= stage_epoch and epoch < stage_epoch * 2:
                
                left_global, right_global, left_global_loss, right_global_loss = self.global_visual_transformation(left_org, right_org, left)
                left_local, right_local, left_local_loss, right_local_loss = self.local_visual_transformation(left_org, right_org, left, right)

                refimg_fea_org_global_local     = self.feature_extraction(torch.cat([left, left_global, left_local], dim=0))
                targetimg_fea_org_global_local  = self.feature_extraction(torch.cat([right, right_global, right_local], dim=0))
                refimg_fea_org, targetimg_fea_org = refimg_fea_org_global_local[0:b], targetimg_fea_org_global_local[0:b]
                refimg_fea_global, targetimg_fea_global = refimg_fea_org_global_local[b:2*b], targetimg_fea_org_global_local[b:2*b]
                refimg_fea_local, targetimg_fea_local = refimg_fea_org_global_local[2*b:], targetimg_fea_org_global_local[2*b:]

                # the caculation of distance loss by minimizing cross-domain feature inconsistency between original feature and transformed feature
                loss_dist = (torch.mean((refimg_fea_global - refimg_fea_org).pow(2)) + torch.mean((targetimg_fea_global - targetimg_fea_org).pow(2)) 
                           + torch.mean((refimg_fea_local - refimg_fea_org).pow(2)) + torch.mean((targetimg_fea_local - targetimg_fea_org).pow(2))) / 2
                # the caculation of cross-entropy loss and discrepency loss
                loss_CE, loss_disc = self.max_discrepency_loss_stage2(left, right, left_global_loss, right_global_loss, left_local_loss, right_local_loss, b)

                loss_hvt_stage2 = self.args.lambda_1 * loss_dist + self.args.lambda_2 * loss_disc + self.args.lambda_3 * loss_CE

                disp_ests_org = []
                disp_ests_global = []
                disp_ests_local = []
                disp_ests_org_global_local = self.cost_volume_and_disparity_regression(refimg_fea_org_global_local, targetimg_fea_org_global_local, left)
                for disp_ests_tmp in disp_ests_org_global_local:
                    disp_ests_org.append(disp_ests_tmp[0:b])
                    disp_ests_global.append(disp_ests_tmp[b:2*b])
                    disp_ests_local.append(disp_ests_tmp[2*b:])

                return [disp_ests_org, disp_ests_global, disp_ests_local], loss_hvt_stage2

            if epoch >= stage_epoch * 2:
                left_global, right_global, left_global_loss, right_global_loss = self.global_visual_transformation(left_org, right_org, left)
                left_local, right_local, left_local_loss, right_local_loss = self.local_visual_transformation(left_org, right_org, left, right)
                left_pixel, right_pixel = self.pixel_visual_transformation(left, right)

                refimg_fea_org_global_local_pixel     = self.feature_extraction(torch.cat([left, left_global, left_local, left_pixel], dim=0))
                targetimg_fea_org_global_local_pixel  = self.feature_extraction(torch.cat([right, right_global, right_local, right_pixel], dim=0))
                refimg_fea_org, targetimg_fea_org = refimg_fea_org_global_local_pixel[0:b], targetimg_fea_org_global_local_pixel[0:b]
                refimg_fea_global, targetimg_fea_global = refimg_fea_org_global_local_pixel[b:2*b], targetimg_fea_org_global_local_pixel[b:2*b]
                refimg_fea_local, targetimg_fea_local = refimg_fea_org_global_local_pixel[2*b:3*b], targetimg_fea_org_global_local_pixel[2*b:3*b]
                refimg_fea_pixel, targetimg_fea_pixel = refimg_fea_org_global_local_pixel[3*b:], targetimg_fea_org_global_local_pixel[3*b:]

                # the caculation of distance loss by minimizing cross-domain feature inconsistency between original feature and transformed feature
                loss_dist = (torch.mean((refimg_fea_global - refimg_fea_org).pow(2)) + torch.mean((targetimg_fea_global - targetimg_fea_org).pow(2)) 
                           + torch.mean((refimg_fea_local - refimg_fea_org).pow(2)) + torch.mean((targetimg_fea_local - targetimg_fea_org).pow(2))
                           + torch.mean((refimg_fea_pixel - refimg_fea_org).pow(2)) + torch.mean((targetimg_fea_pixel - targetimg_fea_org).pow(2))) / 3
                # the caculation of cross-entropy loss and discrepency loss
                loss_CE, loss_disc = self.max_discrepency_loss_stage3(left, right, left_global_loss, right_global_loss, left_local_loss, right_local_loss, left_pixel, right_pixel, b)

                loss_hvt_stage3 = self.args.lambda_1 * loss_dist + self.args.lambda_2 * loss_disc + self.args.lambda_3 * loss_CE

                disp_ests_org = []
                disp_ests_global = []
                disp_ests_local = []
                disp_ests_pixel = []
                disp_ests_org_global_local_pixel = self.cost_volume_and_disparity_regression(refimg_fea_org_global_local_pixel, targetimg_fea_org_global_local_pixel, left)
                for disp_ests_tmp in disp_ests_org_global_local_pixel:
                    disp_ests_org.append(disp_ests_tmp[0:b])
                    disp_ests_global.append(disp_ests_tmp[b:2*b])
                    disp_ests_local.append(disp_ests_tmp[2*b:3*b])
                    disp_ests_pixel.append(disp_ests_tmp[3*b:])

                return [disp_ests_org, disp_ests_global, disp_ests_local, disp_ests_pixel], loss_hvt_stage3
    
