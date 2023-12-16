from __future__ import print_function

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import SecenFlowLoader as DA
from dataloader import listflowfile as lt
from dataloader import ETH3D_loader as et
from dataloader import KITTI2012loader as kt2012
from dataloader import KITTIloader as kt
from dataloader import middlebury_loader as mb
from models import *

from submission import dg_test

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default="/data/tychang/SceneFlow_archive/",
                    help='datapath')
parser.add_argument('--epochs', type=int, default=45,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default="/home/tychang/github_sm_hvt/psmnet_hvt/output",
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

############## the argument of our HVT method #####################
parser.add_argument('--bs_train', type=int, default=8,
                    help='batch size of training stage (default: 8)')
parser.add_argument('--bs_val', type=int, default=1,
                    help='batch size of testing stage (default: 1)')
parser.add_argument('--num_epoch', type=int, default=45,
                    help='the number of epoch')
parser.add_argument('--logfile', type=str, default='/home/tychang/github_sm_hvt/psmnet_hvt/output/hvt_psmnet_bn.txt',
                    help='the domain generalization evaluation results on four realistic datasets')
parser.add_argument('--res18', type=str, default='/home/tychang/github_sm_hvt/psmnet_gLR_res18_final/resnet18-5c106cde.pth',
                    help='the pretrained model of resnet18')

parser.add_argument('--mu', type=float, default=0.1,
                    help='the value of hyper-parameter μ (default: 0.1)')
parser.add_argument('--beta', type=float, default=0.15,
                    help='the value of hyper-parameter β (default: 0.15)')
parser.add_argument('--lambda_1', type=float, default=1.0,
                    help='the value of hyper-parameter λ1  (default: 1.0)')
parser.add_argument('--lambda_2', type=float, default=0.5,
                    help='the value of hyper-parameter λ2  (default: 0.5)')
parser.add_argument('--lambda_3', type=float, default=0.5,
                    help='the value of hyper-parameter λ3  (default: 0.5)')
parser.add_argument('--num_patch', type=int, default=16,
                    help='the number of patches in local transformation (default: 16)')

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= args.bs_train, shuffle= True, num_workers= 8, drop_last=True)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= args.bs_val, shuffle= False, num_workers= 4, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp, args)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel, map_location=torch.device('cpu'))
    model.load_state_dict(pretrain_dict['state_dict'], strict=False)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

def disp2distribute(disp_gt, max_disp, b=2):
    disp_gt = disp_gt.unsqueeze(1)
    disp_range = torch.arange(0, max_disp).view(1, -1, 1, 1).float().cuda()
    gt_distribute = torch.exp(-torch.abs(disp_range - disp_gt) / b)
    gt_distribute = gt_distribute / (torch.sum(gt_distribute, dim=1, keepdim=True) + 1e-8)
    return gt_distribute

def CEloss(disp_gt, max_disp, gt_distribute, pred_distribute):
    mask = (disp_gt > 0) & (disp_gt < max_disp)

    pred_distribute = torch.log(pred_distribute + 1e-8)

    ce_loss = torch.sum(-gt_distribute * pred_distribute, dim=1)
    ce_loss = torch.mean(ce_loss[mask])
    return ce_loss

def train(imgL,imgR, disp_L, imgL_org, imgR_org, batch_idx, epoch):
        model.train()

        if args.cuda:
            imgL, imgR, disp_true, imgL_org, imgR_org = imgL.cuda(), imgR.cuda(), disp_L.cuda(), imgL_org.cuda(), imgR_org.cuda()

        #---------
        mask = disp_true < args.maxdisp
        mask.detach_()
        #----
        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            outputs_list, loss_hvt = model(imgL, imgR, imgL_org, imgR_org, epoch=epoch, training=True)
            loss_disp = 0
            for output in outputs_list:
                output1 = torch.squeeze(output[0],1)
                output2 = torch.squeeze(output[1],1)
                output3 = torch.squeeze(output[2],1)
                loss_disp_tmp = (0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)) / (0.5+0.7+1) 
                loss_disp += loss_disp_tmp
            loss = loss_disp / len(outputs_list) + loss_hvt.sum() / loss_hvt.shape[0]
            
        elif args.model == 'basic':
            output = model(imgL,imgR)
            output = torch.squeeze(output,1)
            loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()

        return loss.data

def test(imgL,imgR, disp_true, batch_idx):

        model.eval()
  
        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
        #---------
        mask = disp_true < 192
        #----

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16       
            top_pad = (times+1)*16 -imgL.shape[2]
        else:
            top_pad = 0

        if imgL.shape[3] % 16 != 0:
            times = imgL.shape[3]//16                       
            right_pad = (times+1)*16-imgL.shape[3]
        else:
            right_pad = 0  

        imgL = F.pad(imgL,(0,right_pad, top_pad,0))
        imgR = F.pad(imgR,(0,right_pad, top_pad,0))

        with torch.no_grad():
            output3 = model(imgL,imgR, imgL,imgR, batch_idx, training=False)
            output3 = torch.squeeze(output3)
        
        if top_pad !=0:
            img = output3[:,top_pad:,:]
        else:
            img = output3

        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = F.l1_loss(img[mask],disp_true[mask]) #torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

        return loss.data.cpu()

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    if epoch < 10:
        lr = lr 
    if epoch >= 10 and epoch < 20:
        lr = lr * 0.5
    if epoch >= 20 and epoch < 30:
        lr = lr * 0.5 * 0.5
    if epoch >= 30 and epoch < 40:
        lr = lr * 0.5 * 0.5 * 0.5
    if epoch>=40:
        lr = lr * 0.5 * 0.5 * 0.5 * 0.5
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_full_time = time.time()
    for epoch in range(0, args.epochs):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, imgL_org, imgR_org) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop,imgR_crop, disp_crop_L, imgL_org, imgR_org, batch_idx, epoch)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss


        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

        #SAVE
        # total_train_loss = 100
        savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(TrainImgLoader),
        }, savefilename)

        model_test = stackhourglass_test(args.maxdisp)
        model_test = nn.DataParallel(model_test)
        model_test.cuda()
        state_dict = torch.load(savefilename)
        model_test.load_state_dict(state_dict['state_dict'], strict=False)
        # log_file = os.path.join(args.savemodel, 'log-psm-dcml-irmv1-test.txt') 
        with open(args.logfile, 'a') as f:
            print('this is the dg test result of epoch {}\n'.format(epoch), file=f)
        dg_test(model_test, args.logfile, test_left_img_mb, test_right_img_mb, train_gt_mb, test_name='md')
        dg_test(model_test, args.logfile, test_left_img_eth, test_right_img_eth, all_disp_eth, test_name='eth', all_mask=all_mask_eth)
        dg_test(model_test, args.logfile, test_left_img_k12, test_right_img_k12, test_ldisp_k12, test_name='kitti15')
        dg_test(model_test, args.logfile, test_left_img_k15, test_right_img_k15, test_ldisp_k15, test_name='kitti12')

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

    # #------------- TEST ------------------------------------------------------------
    # total_test_loss = 0
    # for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
    #     test_loss = test(imgL,imgR, disp_L, batch_idx)
    #     print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
    #     total_test_loss += test_loss

    # print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
    # #----------------------------------------------------------------------------------
    # #SAVE test information
    # savefilename = args.savemodel+'testinformation.tar'
    # torch.save({
    # 'test_loss': total_test_loss/len(TestImgLoader),
    # }, savefilename)


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
    main()
    
