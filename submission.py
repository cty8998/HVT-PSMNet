from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
from PIL import Image
from dataloader import KITTIloader as kt
from dataloader import KITTI2012loader as kt2012
from dataloader import ETH3D_loader as et
from dataloader import middlebury_loader as mb
from torch.utils.data import DataLoader
from dataloader import readpfm as rp

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--test_name', default='kitti12',
                    help='KITTI 2012: kitti12; KITTI 2015: kitti15; Middlebury: md; ETH3D: eth')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--logfile', type=str, default='/home/tychang/github_sm_hvt/psmnet_hvt/output/hvt_psmnet_bn.txt',
                    help='the domain generalization evaluation results on four realistic datasets')
args = parser.parse_args()


def test(model,imgL,imgR):
    model.eval()

    imgL = imgL.cuda()
    imgR = imgR.cuda()     

    with torch.no_grad():
        output = model(imgL,imgR)
    output = torch.squeeze(output).data.cpu().numpy()
    return output

def dg_test(model, log_file, test_left_img, test_right_img, test_ldisp=None, test_name=None, all_mask=None):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])  

    op = 0
    mae = 0
    pred_op = 0
    pred_mae = 0
    D1 = 0
    for inx in range(len(test_left_img)):
        imgL_o = Image.open(test_left_img[inx]).convert('RGB')
        imgR_o = Image.open(test_right_img[inx]).convert('RGB')

        w, h = imgL_o.size

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)         

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        start_time = time.time()
        pred_disp = test(model,imgL,imgR)
        # print('time = %.2f' %(time.time() - start_time))
        # print("第{}张".format(inx))

        if top_pad !=0 or right_pad != 0:
            img = pred_disp[top_pad:,:w]
        else:
            img = pred_disp
        if test_name=='sceneflow':
            predict_np = img
            disp_gt, _ = rp.readPFM(test_ldisp[inx])
            disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
            disp_gt[disp_gt == np.inf] = 0
            mask = (disp_gt > 0) & (disp_gt < 192)
            error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
            pred_mae_ = np.mean(error[mask])
            # print(pred_mae_)
            mae += pred_mae_
            if math.isnan(mae):
                a = 0

        if test_name=='md':
            
            pred_np = img
            disp_gt, _ = rp.readPFM(test_ldisp[inx])
            disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)

            disp_gt[disp_gt == np.inf] = 0
            occ_mask = Image.open(test_ldisp[inx].replace('disp0GT.pfm', 'mask0nocc.png')).convert('L')
            occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32)

            mask = (disp_gt <= 0) | (occ_mask != 255) | (disp_gt >= args.maxdisp)

            error = np.abs(pred_np - disp_gt)
            error[mask] = 0

            if inx in [6, 8, 9, 12, 14]:
                k = 1
            else:
                k = 1
            op_ = np.sum(error > 2.0) / (w * h - np.sum(mask)) * k
            # print(op_)
            mae_ = np.sum(error) / (w * h - np.sum(mask)) * k
            # print(mae_)
            op += op_
            mae += mae_
        if test_name=='eth':
            if inx == 7:
                a = 0
            predict_np = img
            disp_gt, _ = rp.readPFM(test_ldisp[inx])
            disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
            disp_gt[disp_gt == np.inf] = 0

            occ_mask = np.ascontiguousarray(Image.open(all_mask[inx]))
            predict_np = img
            op_thresh = 1
            mask = (disp_gt > 0) & (occ_mask == 255)
            error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

            pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
            pred_op_ = np.sum(pred_error > op_thresh) / np.sum(mask)
            # print(pred_op_)
            pred_op += pred_op_
            pred_mae_ = np.mean(pred_error[mask])
            # print(pred_mae_)
            pred_mae += pred_mae_
        if test_name=='kitti12' or test_name=='kitti15':
            disp_gt = Image.open(test_ldisp[inx])
            disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32) / 256
            predict_np = img

            op_thresh = 3
            mask = (disp_gt > 0) & (disp_gt < args.maxdisp)
            error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

            D1_mask = (np.abs(predict_np[mask] - disp_gt[mask]) > op_thresh) & (np.abs(predict_np[mask] - disp_gt[mask]) / np.abs(disp_gt[mask]) > 0.05)
            D1_ = np.mean(D1_mask)
            # print(D1_)
            D1 += D1_
            pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
            pred_op_ = np.sum((pred_error > op_thresh)) / np.sum(mask)
            # print(pred_op_)
            pred_op += pred_op_
            pred_mae_ = np.mean(pred_error[mask])
            # print(pred_mae_)
            pred_mae += np.mean(pred_mae_)
    if test_name=='sceneflow':
        print('********{}********'.format(test_name))
        print('epe:{}'.format(mae / len(test_left_img)))
    if test_name=='md':
        with open(log_file, 'a') as f:
            print('********{}********'.format(test_name), file=f)
            print('2px:{}'.format(op / 15 * 100), file=f)
            print('epe:{}'.format(mae / 15), file=f)
    if test_name=='eth':
        with open(log_file, 'a') as f:
            print('********{}********'.format(test_name), file=f)
            print('1px:{}'.format(pred_op / len(test_left_img) * 100), file=f)
            print('epe:{}'.format(pred_mae / len(test_left_img)), file=f)
    if test_name=='kitti12' or test_name=='kitti15':
        with open(log_file, 'a') as f:
            print('********{}********'.format(test_name), file=f)
            print('3px:{}'.format(pred_op / len(test_left_img) * 100), file=f)
            print('D1:{}'.format(D1 / len(test_left_img) * 100), file=f)
            print('epe:{}'.format(pred_mae / len(test_left_img)), file=f)



if __name__ == '__main__':
    train_limg_mb, train_rimg_mb, train_gt_mb, test_limg_mb, test_rimg_mb = mb.mb_loader("/home/tychang/test_data//MiddEval3", res='H')
    test_left_img_mb, test_right_img_mb = train_limg_mb, train_rimg_mb

    all_limg_eth, all_rimg_eth, all_disp_eth, all_mask_eth = et.et_loader("/home/tychang/test_data//ETH3D")
    test_left_img_eth, test_right_img_eth = all_limg_eth, all_rimg_eth

    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt.kt_loader("/home/tychang/test_data//KITTI_stereo/kitti_2015/data_scene_flow/training/")
    test_left_img_k12, test_right_img_k12 = all_limg + test_limg, all_rimg + test_rimg
    test_ldisp_k12 = all_ldisp + test_ldisp
    
    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2012.kt2012_loader("/home/tychang/test_data//KITTI_stereo/kitti_2012/data_stereo_flow/training/")
    test_left_img_k15, test_right_img_k15 = all_limg + test_limg, all_rimg + test_rimg
    test_ldisp_k15 = all_ldisp + test_ldisp

    model_test = stackhourglass_test(args.maxdisp)
    model_test = nn.DataParallel(model_test)
    model_test.cuda()
    state_dict = torch.load(args.loadmodel)
    model_test.load_state_dict(state_dict['state_dict'], strict=False)
    
    dg_test(model_test, args.logfile, test_left_img_mb, test_right_img_mb, train_gt_mb, test_name=args.test_name)
    # dg_test(model_test, args.logfile, test_left_img_eth, test_right_img_eth, all_disp_eth, test_name='eth', all_mask=all_mask_eth)
    # dg_test(model_test, args.logfile, test_left_img_k12, test_right_img_k12, test_ldisp_k12, test_name='kitti15')
    # dg_test(model_test, args.logfile, test_left_img_k15, test_right_img_k15, test_ldisp_k15, test_name='kitti12')




