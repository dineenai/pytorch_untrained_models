#  Addapted from main.py untrained resnetx

# NB TRY IMPORTIG THE RESNET 50 FROM THE NEW SCRIPT!!!!!!!!!
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import cv2 as cv
import argparse
# from torchvision import models, transforms
from torchvision import transforms
import resnet_conv1_21 as models

import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils





model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')



def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))








def main():
    args = parser.parse_args()


# def main_worker(gpu, ngpus_per_node, args):

    global best_acc1

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(NONE) #Do not need?

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            print(model) #NB Prints ResNet with Kernel Size 7x7!!!!! NOT out Epoch!
            model.load_state_dict(checkpoint['state_dict'])
            # model = model.load_state_dict(checkpoint['state_dict']) #TRY - still printing untrained rfs
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # print(model)





# https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch
    child_counter = 0
    for child in model.children():
        if child_counter < 6:
            for param in child.parameters():
                param.requires_grad = False
        elif child_counter == 6:
            children_of_child_counter = 0
            for children_of_child in child.children():
                if children_of_child_counter < 1:
                    for param in children_of_child.parameters():
                        param.requires_grad = False
                else:
                    children_of_child_counter += 1

        else:
            print("child ",child_counter," was not frozen")
        child_counter += 1







    model.eval()
    
    # Receptive Fields
    model_weights = [] # we will save the conv layer weights in this list
    conv_layers = [] # we will save the 49 conv layers in this list
    # get all the model children as list
    # model_children = list(model.children())

    # Is this showing the untrained model or the trained model? - looks untrained!
    # model_children = list(models.resnet50().children())
   
   


    # print(model)
    # print("Model Children")
    
    # print(len(model_children))

    child_counter = 0
    for child in model.children():
        print(" child", child_counter, "is:")
        print(child)
        child_counter += 1

    model_children = list(model.children())



    # get the kernels from the first layer
    # as per the name of the layer
    kernels = model.first_conv_layer.weight.detach().clone()

    #check size for sanity check
    print(kernels.size())

    # normalize to (0,1) range so that matplotlib
    # can plot them
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    filter_img = torchvision.utils.make_grid(kernels, nrow = 12)
    # change ordering since matplotlib requires images to 
    # be (H, W, C)
    plt.imshow(filter_img.permute(1, 2, 0))

    # You can directly save the image as well using
    img = save_image(kernels, 'encoder_conv1_filters.png' ,nrow = 12)


    # layer = 1
    # filter = model.features[layer].weight.data.clone()
    # visTensor(filter, ch=0, allkernels=False)

    # plt.axis('off')
    # plt.ioff()
    # plt.show()
    # plt.savefig('/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/TEXT_RF.png')







#     model_children = list(model.children())
#     Traceback (most recent call last):
#   File "vis_rf.py", line 154, in <module>
#     main()
#   File "vis_rf.py", line 145, in main
#     for i, filter in enumerate(model_weights[0]):
# IndexError: list index out of range


    # model_children = list(model.resnet50().children())
    # model_children = list(model.children())
    # print("model_children")
    # print("model_children")
    # print("model_children")
    # print(model_children)
    # print(len(model_children)) #1

    # model_children = list(model.resnet50().children())
    #   File "vis_rf.py", line 108, in main
    # model_children = list(model.resnet50().children())
    #  File "/opt/anaconda3/envs/blurry_vision/lib/python3.7/site-packages/torch/nn/modules/module.py", line 779, in __getattr__
    #   type(self).__name__, name))
    #   torch.nn.modules.module.ModuleAttributeError: 'DataParallel' object has no attribute 'resnet50'
    # print(len((model.resnet50().children())))


    print(model_children)
    # https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
    counter = 0 
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")

    # take a look at the conv layers and the respective weights
    for weight, conv in zip(model_weights, conv_layers):
        # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
        print(f"CONV: {conv} ====> SHAPE: {weight.shape}")



    # visualize the first conv layer filters
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
        plt.savefig('/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/21x21FILTER_TEXT_resnet50_conv1_21.png')
    # plt.show()


if __name__ == '__main__':
    main()
    