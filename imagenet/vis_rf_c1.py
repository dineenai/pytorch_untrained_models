# https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim
import cv2 as cv
import argparse
from torchvision import models, transforms

# # WORKS
# # load the model
# model = models.resnet50(pretrained=False)
# print(model)
# model_weights = [] # we will save the conv layer weights in this list
# conv_layers = [] # we will save the 49 conv layers in this list
# # get all the model children as list
# model_children = list(model.children())


# RESUME FOR CHECKPOINT!!!!! - add args at end
#  Add Checlpoint Loader ONCE code is working!!!!! ####
# # optionally resume from a checkpoint
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')

# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# args = parser.parse_args()

cpkt_path="/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/model_best_supervised_resnet50_conv1_21_gauss_0_for_60_epoch.pth.tar"

model = models.resnet50(pretrained=False)

# print("=> creating model '{}'".format(args.arch))
# model = models.__dict__[args.arch]()

# model = models.__dict__[resnet50]()

# LOADS CHECKPINT BUT NOT PRINTING CORRECT Model!!!!!!!!! ######

lr = 0.1 #Add parser args.lr
weight_decay = 1e-4 #args.weight_decay
momentum = 0.9 #args.momentum
optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

if os.path.isfile(cpkt_path):
    print("=> loading checkpoint '{}'".format(cpkt_path))
    checkpoint = torch.load(cpkt_path)
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'], strict=False) #https://stackoverflow.com/questions/54058256/runtimeerror-errors-in-loading-state-dict-for-resnet
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(cpkt_path, checkpoint['epoch']))
    # print(model.load_state_dict(checkpoint['state_dict']))
else:
    print("=> no checkpoint found at '{}'".format(args.resume))

# model.state_dict()
model_weights = [] # we will save the conv layer weights in this list
conv_layers = [] # we will save the 49 conv layers in this list
# get all the model children as list
model_children = list(model.children())

# print(model.state_dict())
# print(model)

#     # if args.gpu is None:
    
#     # else:
#     #     # Map model to be loaded to specified single gpu.
#     #     loc = 'cuda:{}'.format(args.gpu)
#     #     checkpoint = torch.load(args.resume, map_location=loc)
#     # args.start_epoch = checkpoint['epoch']
#     # start_epoch = checkpoint['epoch']
#     # best_acc1 = checkpoint['best_acc1']
#     # if args.gpu is not None:
#     #     # best_acc1 may be from a checkpoint from a different GPU
#     #     best_acc1 = best_acc1.to(args.gpu)
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print("=> loaded checkpoint '{}' (epoch {})"
#           .format(cpkt_path, checkpoint['epoch']))
# else:
#     print("=> no checkpoint found at '{}'".format(args.resume))



# https://stackoverflow.com/questions/54058256/runtimeerror-errors-in-loading-state-dict-for-resnet
# # instantiate model
# conv = ConvModel()

# # load weights if they haven't been loaded
# # skip if you're directly importing a pretrained network
# checkpoint = torch.load('model_weights.pt')
# conv.load_state_dict(checkpoint)


# # get the kernels from the first layer
# # as per the name of the layer
# kernels = conv.first_conv_layer.weight.detach().clone()

# #check size for sanity check
# print(kernels.size())



# # WORKS but printing wrong model!!!
print(model)

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
    plt.savefig('/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/FILTER_TEXT_resnet50_conv1_21.png')
# plt.show()






























# #https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import utils

# def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
#     n,c,w,h = tensor.shape

#     if allkernels: tensor = tensor.view(n*c, -1, w, h)
#     elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

#     rows = np.min((tensor.shape[0] // nrow + 1, 64))    
#     grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
#     plt.figure( figsize=(nrow,rows) )
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))


# if __name__ == "__main__":
#     layer = 1
#     filter = model.features[layer].weight.data.clone()
#     visTensor(filter, ch=0, allkernels=False)

#     plt.axis('off')
#     plt.ioff()
#     plt.show()




# # instantiate model
# conv = ConvModel()

# # load weights if they haven't been loaded
# # skip if you're directly importing a pretrained network
# checkpoint = torch.load('model_weights.pt')
# conv.load_state_dict(checkpoint)


# # get the kernels from the first layer
# # as per the name of the layer
# kernels = conv.first_conv_layer.weight.detach().clone()

# #check size for sanity check
# print(kernels.size())

# # normalize to (0,1) range so that matplotlib
# # can plot them
# kernels = kernels - kernels.min()
# kernels = kernels / kernels.max()
# filter_img = torchvision.utils.make_grid(kernels, nrow = 12)
# # change ordering since matplotlib requires images to 
# # be (H, W, C)
# plt.imshow(filter_img.permute(1, 2, 0))

# # You can directly save the image as well using
# img = save_image(kernels, 'encoder_conv1_filters.png' ,nrow = 12)
