
import network
import os
import argparse
import numpy as np


import torch
import torch.nn as nn

from torchvision.models import *
from torchsummary import summary

import pyprof
import torch.cuda.profiler as profiler
from ptflops import get_model_complexity_info

from thop import profile


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # Set the GPU 2 to use


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/root/daehyeonchoi/POSTECH-CSED539/POSTECH-CSED539/datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    
    parser.add_argument("--wavelets", type=bool, default=True,
                        help='using wavelet transform (i.e. DeepLabV3PlusW Model)')
    
    parser.add_argument("--upsampler", type=str, default='defup', choices=['nn', 'bilinear', 'carafev1', 
                                                                              'carafev2', 'defup', 'sapa'],
                        help='select feature upsampler options')


    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30000,
                        help="total iterations (default: 30k)")
    parser.add_argument("--total_epochs", type=int, default=20,
                        help="total epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=True,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=8,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--ckpt", default="pth/best_deeplabv3plus_resnet101_voc_os16_carafe.pth", type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='3',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=200, # about 12800 images for batch 64 
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012_aug',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')
    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def check_flops_and_params(model, device, type):
    
    if type == 'no':
        x_ll = torch.randn(1, 3, 512, 512).to(device)
        with torch.cuda.device(device):
            flops, params = profile(model=model, inputs = (x_ll,))
            print('{:<30}  {:<8}'.format('Computational complexity: ', flops/(2**30)))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))  
            
    elif type == 'nn' or type == 'bilinear':
        x_ll = torch.randn(1, 3, 256, 256).to(device)
        hfs = torch.randn(1, 9, 256, 256).to(device)
        with torch.cuda.device(device):
            flops, params = profile(model=model, inputs = (x_ll, ))
            print('{:<30}  {:<8}'.format('Computational complexity: ', flops/(2**30)))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))  
            
    else: # Feature Upsampler with Wavelet
        x_ll = torch.randn(1, 3, 256, 256).to(device)
        hfs = torch.randn(1, 9, 256, 256).to(device)
        with torch.cuda.device(device):
            flops, params = profile(model=model, inputs = (x_ll, hfs))
            print('{:<30}  {:<8}'.format('Computational complexity: ', flops/(2**30)))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))  
             
   
def check_times(model, device, type):

    H, W = 512, 512 
    
    if type == 'no': pass
    else: H, W = H//2, W//2
    x_ll = torch.randn((4, 3, H, W)).to(device)
    hfs = torch.randn((4, 9, H, W)).to(device)
    
    repetitions = 300
    
    timings = np.zeros((repetitions,1))
    
    if type == 'no': 
        
        for _ in range(10):
            features = model.backbone(x_ll)
            _ = model.classifier(features)
        
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)   # Time evaluation
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                input_shape = x_ll.shape[-2:]
                features = model.backbone(x_ll)
                x = model.classifier(features)
                x = torch.nn.functional.interpolate(x, size = input_shape, mode = 'bilinear', align_corners = False) 

                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
                
        print('{:<30}  {:<8}'.format('Mean Infrenece Time: ', np.mean(timings)))
        return 
    
    elif type == 'nn' or type == 'bilinear':
        
        for _ in range(10):
            features = model.backbone(x_ll)
            _ = model.classifier(features)
        
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)   # Time evaluation
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                input_shape = x_ll.shape[-2:]
                features = model.backbone(x_ll)  
                x = model.classifier(features)
                upsample_size = tuple(size * 2 for size in input_shape)
                x = torch.nn.functional.interpolate(x, size = upsample_size, mode = 'bilinear', align_corners = False) 

                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
                
        print('{:<30}  {:<8}'.format('Mean Infrenece Time: ', np.mean(timings)))
        return 
    
    else: 
        
        for _ in range(10):
            features = model.backbone(x_ll)
            _ = model.classifier(features, hfs)
        
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)   # Time evaluation
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                input_shape = x_ll.shape[-2:]
                features = model.backbone(x_ll) 
                x = model.classifier(features, hfs)
                upsample_size = tuple(size * 2 for size in input_shape)
                x = torch.nn.functional.interpolate(x, size = upsample_size, mode = 'bilinear', align_corners = False) 

                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
                
        print('{:<30}  {:<8}'.format('Mean Infrenece Time: ', np.mean(timings)))
        return 


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    
    
    upsampler_list = ['nn', 'bilinear', 'carafev1','carafev2', 'defup', 'no'] # check your upsampler and set in 'opts.upsampler'k
    opts.upsampler = 'no'
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](upsampler=opts.upsampler, num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
        
    model.to(device)
    
    check_times(model, device, opts.upsampler)
    check_flops_and_params(model, device, opts.upsampler)

if __name__ == '__main__':
    main()
