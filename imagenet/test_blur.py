
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
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.GaussianBlur(9, sigma=(4, 4)),
            transforms.functional.gaussian_blur(), #module 'torchvision.transforms.functional' has no attribute 'gaussian_blur'
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
            transforms.ToTensor(),
            normalize,
        ]))
    
    out_dir = "/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/test_out/"
    
    # for i in range(test_dataset): #'ImageFolder' object cannot be interpreted as an integer
    #     print(i)   


        #out_dir = output_dir.joinpath(rel_path.parent) #Better than bellow!
        # out_filename = 
        # output_name = out_dir.joinpath(out_filename)
    save_image(i, out_dir) #default image padding is 2.

    #Print/ Output Image Before and after blur - how - save it somehow?
    


    # transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))

    #torchvision.transforms.functional.gaussian_blur(img: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None) → torch.Tensor

    test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    for i in test_loader:
        print(images.shape)
        img1 = images[0] #torch.Size([3,28,28]
        # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
        save_image(img1, 'img1.png')

    for idx in test_loader:
        print(idx)   
    for idx,val in enumerate(['a','b','c']):
        print('index of ' + val + ': ' + str(idx))
    
    # for idx, name in enumerate(presidents, start=1):
    # print("President {}: {}".format(num, name))

    print(len(test_loader))

    # presidents = ["Washington", "Adams", "Jefferson", "Madison", "Monroe", "Adams", "Jackson"]
    
    # for num, name in enumerate(presidents, start=1):
    #     print("President {}: {}".format(num, name))

    # for data in test_loader:
    #     # print(data)
    #     # save_image
    # dataset = MyDataset(..., transform=transform)

    # for idx, data, target in enumerate(dataset):
    #     torch.save(data, 'data_drive_path{}'.format(idx))
    #     torch.save(target, ...

    # #Does NOT work!
    # for idx, data, target in enumerate(test_loader):
    #     torch.save(data, '~/blurry_vision/pytorch_untrained_models/imagenet/test_out/'.format(idx))
    


    # torch.save(target, ...
    
    # for i in test_loader:
    #     torch.save(data, '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/test_out')
    #     # torch.save(target, ...


    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
        # batch_size=args.batch_size, shuffle=False,
        # num_workers=args.workers, pin_memory=True)


if __name__ == '__main__':
    main()