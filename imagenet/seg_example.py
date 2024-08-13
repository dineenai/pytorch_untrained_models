from segmentation import segment_white
import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(description='PyTorch Segmentation Code')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-s', '--segment', default='rm_bg', type=str,
                    metavar='N',
                    help='TEXT')

args = parser.parse_args()

valdir = os.path.join(args.data, 'val') 

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
#dataloader defined as per pytorch code
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)


for i, (images, target) in enumerate(val_loader):
    with torch.no_grad(): 

        input_var = images.cuda()
        # input_var, label = input_tensor[0].cuda(),input_tensor[2][0]
 
        if args.segment == 'rm_bg':
            input_var = segment_white(input_var)
            input_var = input_var.unsqueeze(0)









# # From CMC    train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
#         transforms.RandomHorizontalFlip(),
#         color_transfer,
#         transforms.ToTensor(),
#         normalize,
#     ])
#     train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
#     train_sampler = None

#     # train loader
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#         num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)