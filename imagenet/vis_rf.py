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
# import collections #OrderedDict
from collections import OrderedDict

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

            state_dict_new = OrderedDict()
            # for k, v in checkpoint.items():
            for k, v in checkpoint['state_dict'].items():
                name = k.replace(".module", '') # remove 'module.' of dataparallel
                state_dict_new[name]=v


            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            # args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']

            # print("args.start_epoch")
            # print(args.start_epoch)
            # print("best_acc1")
            # print(best_acc1)

            model.load_state_dict(checkpoint['state_dict'])
            # model.load_state_dict(state_dict_new)
            # model.load_state_dict(checkpoint_new['state_dict'])
            # model.load_state_dict(checkpoint['state_dict'].items())

            print(model)
            # print("checkpoint.keys()")
            # print(checkpoint.keys())

            # print("checkpoint['state_dict']")
            # print(checkpoint['state_dict'])


# ORINGINAL
            # print(model) #NB Prints ResNet with Kernel Size 7x7!!!!! NOT out Epoch!
            


            # Results in:
            # Total convolutional layers: 0
            # Traceback (most recent call last):
            #   File "vis_rf.py", line 196, in <module>
            #     main()
            #   File "vis_rf.py", line 187, in main
            #     for i, filter in enumerate(model_weights[0]):
            # IndexError: list index out of range

            
            # print("State Dict:")
            # print(state_dict) #NameError: name 'state_dict' is not defined
            # print('str(checkpoint)')
            # print(str(checkpoint))

            # model = torch.load(checkpoint)
            # torch.load(checkpoint)
            # AttributeError: 'dict' object has no attribute 'seek'. You can only torch.load from a file that is seekable
            # Please pre-load the data into a buffer like io.BytesIO and try to load from it instead.

            # model.load_state_dict(checkpoint)
            # Error: Unexpected key(s) in state_dict: "epoch", "arch", "state_dict", "best_acc1", "optimizer"



            # model = model.load_state_dict(checkpoint['state_dict']) #TRY - still printing untrained rfs
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.resume, checkpoint['epoch']))

            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # print(model)



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

    # child_counter = 0
    # for child in model.children():
    #     print(" child", child_counter, "is:")
    #     print(child)
    #     child_counter += 1

    model_children = list(model.children())
    print("model.children()")
    print(model.children())
    print(len(model_children))

#     model_children = list(model.children())
#     Traceback (most recent call last):
#   File "vis_rf.py", line 154, in <module>
#     main()
#   File "vis_rf.py", line 145, in main
#     for i, filter in enumerate(model_weights[0]):
# IndexError: list index out of range


    # model_children = list(model.resnet50().children())
    model_children = list(model.children())
    print("model_children")
    print("model_children")
    print("model_children")
    print(model_children)
    print(len(model_children)) #1

    # model_children = list(model.resnet50().children())
    #   File "vis_rf.py", line 108, in main
    # model_children = list(model.resnet50().children())
    #  File "/opt/anaconda3/envs/blurry_vision/lib/python3.7/site-packages/torch/nn/modules/module.py", line 779, in __getattr__
    #   type(self).__name__, name))
    #   torch.nn.modules.module.ModuleAttributeError: 'DataParallel' object has no attribute 'resnet50'
    # print(len((model.resnet50().children())))



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

cd
if __name__ == '__main__':
    main()