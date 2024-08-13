# Refine this script, needs data path input, saves image to /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/blurred_test_images/blurred_persian_cat.jpg'
#Add kernel and sigma parsers
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms.functional #Added to try use GaussianBlur function
from torchvision.utils import save_image #Added to try save images
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    # Data loading code
    testdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) #What does this do?/Where does it work?
    
    # Data loading code
    testdir = os.path.join(args.data, 'test')
    

    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
            transforms.RandomResizedCrop(100),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(9, sigma=(4, 4)),
            # transforms.functional.gaussian_blur(kernel_size=9, sigma=(4, 4)), #module 'torchvision.transforms.functional' has no attribute 'gaussian_blur'
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
            transforms.ToTensor(),
            normalize,
        ]))

    #Use different valied in i and i - may need an i j loop - for now proves blur is working - sufficient
    for i in range(len(test_dataset)):
        torchvision.utils.save_image(test_dataset[i][i], '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/blurred_test_images/blurred_persian_cat_TEST.jpg', normalize = True)
 
#     # torchvision.utils.save_image(test_dataset, "/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/test_out/persian.jpg")
#     # for i in range(len(test_loader)):
#     #     vutils.save_image(test_loader[i], '%s/test_dataset.png' % "./test_out", normalize = True)
    
#     # https://stackoverflow.com/questions/53570181/error-in-importing-libraries-of-helper-though-helper-is-installed
#     # wget -c https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/intro-to-pytorch/helper.py
#     import helper
#     images = next(iter(test_loader))
    
#     # helper.imshow(images[0], normalize=False)
#     # helper.imshow(images.view(-1));

#     save_image(images[2], out_dir) #default image padding is 2.


if __name__ == '__main__':
    main()