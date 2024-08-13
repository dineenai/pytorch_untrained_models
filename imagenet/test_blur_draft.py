
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
    
    
    # from PIL import Image

    # img = Image.open("/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/test/.")



    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(9, sigma=(4, 4)),
            # transforms.functional.gaussian_blur(kernel_size=9, sigma=(4, 4)), #module 'torchvision.transforms.functional' has no attribute 'gaussian_blur'
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
            transforms.ToTensor(),
            normalize,
        ]))


    # test_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(testdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.GaussianBlur(9, sigma=(4, 4)),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),)




    # vutils.save_image(
    #         data_img, '%s/real_samples.png' % image_dir,
    #         normalize=True)

    # from torch.utils.data import Dataset, DataLoader
    # import utils
    # utils.show_images(test_dataset)

    # test_dataset
    # print(type(test_dataset[1]))
    # print(test_dataset[1])

    # batch_size = self.test_dataset.batch_size
    # with torch.no_grad():
    #         for batch_idx, data in enumerate(self.test_dataset):
    #             vutils.save_image('img_{}.png'.format(batch_idx),
    #                               nrow=batch_size,
    #                               normalize=True) 


    for i in range(len(test_dataset)):
        torchvision.utils.save_image(test_dataset[1][i], '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/blurred_test_images/persian_5.jpg', normalize = True)
 

    # img_PIL = Image.open(r'out_dir/persian_2.jpg')

    out_dir = "/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/test_out/"
    
    # from PIL import Image
    # for i in range(len(test_dataset)):
    #     sample = test_dataset[i]
    #     # img_PIL = Image.open(r'out_dir/persian_2.jpg')
    #     img_PIL = Image.open(r'persian_2.jpg')
    #     img_PIL.show()
    
    
    # # images = datasets.ImageFolder(root=args.input_folder)
    # test_dataset.idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    # create_dirs(out_dir, images.classes)

    # for i in range(len(test_dataset)): #'ImageFolder' object cannot be interpreted as an integer
    #     print(test_dataset[i])   
    #     torchvision.utils.save_image(test_dataset[-1][i], out_dir)


        #out_dir = output_dir.joinpath(rel_path.parent) #Better than bellow!
        # out_filename = 
        # output_name = out_dir.joinpath(out_filename)
 
    import torchvision.utils as vutils
    print(type(test_loader))
    print(test_loader)
    # torchvision.utils.save_image(test_dataset, "/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/test_out/persian.jpg")
    # for i in range(len(test_loader)):
    #     vutils.save_image(test_loader[i], '%s/test_dataset.png' % "./test_out", normalize = True)
    
    # https://stackoverflow.com/questions/53570181/error-in-importing-libraries-of-helper-though-helper-is-installed
    # wget -c https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/intro-to-pytorch/helper.py
    import helper
    images = next(iter(test_loader))
    
    # helper.imshow(images[0], normalize=False)
    # helper.imshow(images.view(-1));

    save_image(images[2], out_dir) #default image padding is 2.

    # #Print/ Output Image Before and after blur - how - save it somehow?
    
# def save_image(tensor, filename, nrow=8, padding=2,
#                normalize=False, range=None, scale_each=False, pad_value=0):
#     """Save a given Tensor into an image file.

#     Args:
#         tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
#             saves the tensor as a grid of images by calling ``make_grid``.
#         **kwargs: Other arguments are documented in ``make_grid``.
#     """
#     from PIL import Image
#     grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
#                      normalize=normalize, range=range, scale_each=scale_each)
#     # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
#     ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#     im = Image.fromarray(ndarr)
#     im.save(filename)



    # # transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))

    # #torchvision.transforms.functional.gaussian_blur(img: torch.Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None) → torch.Tensor

    # test_loader = torch.utils.data.DataLoader(
    # test_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
    # num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    # for i in test_loader:
    #     print(images.shape)
    #     img1 = images[0] #torch.Size([3,28,28]
    #     # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
    #     save_image(img1, 'img1.png')

    # for idx in test_loader:
    #     print(idx)   
    # for idx,val in enumerate(['a','b','c']):
    #     print('index of ' + val + ': ' + str(idx))
    
    # # for idx, name in enumerate(presidents, start=1):
    # # print("President {}: {}".format(num, name))

    # print(len(test_loader))

    # # presidents = ["Washington", "Adams", "Jefferson", "Madison", "Monroe", "Adams", "Jackson"]
    
    # # for num, name in enumerate(presidents, start=1):
    # #     print("President {}: {}".format(num, name))

    # # for data in test_loader:
    # #     # print(data)
    # #     # save_image
    # # dataset = MyDataset(..., transform=transform)

    # # for idx, data, target in enumerate(dataset):
    # #     torch.save(data, 'data_drive_path{}'.format(idx))
    # #     torch.save(target, ...

    # # #Does NOT work!
    # # for idx, data, target in enumerate(test_loader):
    # #     torch.save(data, '~/blurry_vision/pytorch_untrained_models/imagenet/test_out/'.format(idx))
    


    # # torch.save(target, ...
    
    # # for i in test_loader:
    # #     torch.save(data, '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/test_out')
    # #     # torch.save(target, ...


    # # val_loader = torch.utils.data.DataLoader(
    # #     datasets.ImageFolder(valdir, transforms.Compose([
    # #         transforms.Resize(256),
    # #         transforms.CenterCrop(224),
    # #         transforms.ToTensor(),
    # #         normalize,
    # #     ])),
    #     # batch_size=args.batch_size, shuffle=False,
    #     # num_workers=args.workers, pin_memory=True)


if __name__ == '__main__':
    main()