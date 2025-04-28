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
# import resnet_conv1_21 as models
import ntpath #Split path: https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
from pytorch_image_folder_with_file_paths import ImageFolderWithPaths #Print Image Name https://stackoverflow.com/questions/56962318/printing-image-paths-from-the-dataloader-in-pytorch
import probabilities_to_decision
from probabilities_to_decision import ImageNetProbabilitiesTo16ClassesMapping
import helper

import pandas as pd

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model_path', type=str, default=None, help='path to save model') #Add Path to save model from CMC
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_freq', type=int, default=10, help='save frequency') #From CMC - AIM: Save Ckpt.
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')                  
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


parser.add_argument('--save_accuracy_path', default='/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/model_accuracy',
                    type=str, metavar='SAVE_ACCURACY_PATH',
                    help='path to save accuracy of model') 
parser.add_argument('--save_accuracy_file', default='model_accuracy', type=str, metavar='ACCURACY_FILENAME',
                    help='filename to save accuracy of model ')  

# parser.add_argument('--save_test_result_path', default='/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/model_accuracy',
#                     type=str, metavar='SAVE_ACCURACY_PATH',
#                     help='path to save accuracy of model') 

parser.add_argument('--test_result_stimuli_name', default='stimuli', type=str, metavar='STIMULI_NAME',
                    help='filename to save accuracy of model ') 

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# parser.add_argument('--gauss', default=0, type=int, metavar='N',
#                     help='amount of gaussian blur to apply to the train_loader') #Added
parser.add_argument('--gauss', default=0, type=float, metavar='N',
                    help='amount of gaussian blur to apply to the val_loader') #Added 
parser.add_argument('--kernel', default=9, type=int, metavar='N',
                    help='size of kernel for gaussian blur applied to train_loader')

opt = parser.parse_args() #Added from train_CMC.py to facilitate saving ckpts

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
        


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1'] #Removing this allows us to use out added checkpoints!
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Set appropriate resize transform based on dataset
    if args.test_result_stimuli_name == '16_class_IN':
        valdir = os.path.join(args.data, 'style-transfer-preprocessed-512') 
        # To calculate shape bias - already 224x224 so should not actually change the stimulus!
        resize_transform = transforms.CenterCrop(224) 

    elif args.test_result_stimuli_name == 'imagenet_val':
        valdir = os.path.join(args.data, 'val_in_folders') 
        # # To calculate accuracy from validation set
        resize_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])


        
    # Blur functions for MMK
    if args.gauss==0:
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                # transforms.CenterCrop(224),
                resize_transform,
                transforms.ToTensor(),
                normalize,
            ]))
    # If blurring:
    else:
        
        if args.gauss==3:
            gauss = 2
            blur_kernel = 13
            blur_transform = transforms.Compose([
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)), #default kernel size of 9, sigma of 4
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)),
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(1, 1)),
                    ])
            
        elif args.gauss==4:
            gauss = 2
            blur_kernel = 13
            blur_transform = transforms.Compose([
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)), #default kernel size of 9, sigma of 4
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)),
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)),
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)),
                    ])
        
        elif args.gauss==6:
            gauss = 2 
            blur_kernel = 13
            blur_transform = transforms.Compose([
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)), #default kernel size of 9, sigma of 4
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)),
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)),
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)),
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)),
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)),
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)),
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)),
                    transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(gauss, gauss)),
                    ])
            
        elif args.gauss in [0.5, 1, 1.5, 2]:
            blur_transform = transforms.GaussianBlur(args.kernel, sigma=(args.gauss, args.gauss))
        
        else:
            raise ValueError("Invalid value for --gauss. Must be 0, 0.5, 1, 1.5, 2, 3, 4, or 6.")

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                # transforms.CenterCrop(224),
                resize_transform,
                blur_transform, 
                transforms.ToTensor(),
                normalize,
            ]))
        

    print(f'Valdir is {valdir}')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    if args.evaluate:

        if args.test_result_stimuli_name == '16_class_IN':
            shape_bias(val_loader, model, criterion, args)

        elif args.test_result_stimuli_name == 'imagenet_val':
            validate(val_loader, model, criterion, args)
        
        return

    


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            # print("Output: ")
            # print(output) #a tensor

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()



            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        print("TRY:\n* Loss:{losses.avg:.4e}".format(losses=losses))

            #   Loss', ':.4e'
        
        # Added file to save average accuracy following validation, added parsers
        # save_path = '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/model_accuracy/'
        save_path = args.save_accuracy_path
        # name_of_file = raw_input("What is the name of the file: ") #Add Parser to capture network name!
        # name_of_file = 'testfile_model_accuracy'
        name_of_file = args.save_accuracy_file
        # output_accuracy = os.path.join(save_path, name_of_file+".txt") 

        # f = open('model_accuracy.txt', 'a')


        # Try 7/9/22
        output_accuracy_csv = os.path.join(save_path, name_of_file+".csv") 
        # f = open(output_accuracy_csv, "a")
        import pandas as pd
        if os.path.exists(output_accuracy_csv):
            val_accuracy_csv = pd.read_csv(output_accuracy_csv, index_col=0)
        else:
            # Added for bandpass analysis
            if 'butter' in name_of_file:
                val_accuracy_csv = pd.DataFrame(columns=['model_pth','epoch', 'top1acc', 'top5acc', 'loss', 'train_band', 'test_band']) #add column for epoch (perhaps instead of path?)
            else:
                val_accuracy_csv = pd.DataFrame(columns=['model_pth','epoch', 'top1acc', 'top5acc', 'loss', 'val_images_blurred', 'val_blur_sigma']) #add column for epoch (perhaps instead of path?)
        
        # TRY adding Epoch:args.start_epoch 
        test_images_blurred = 0 #make this a binary variable - y or n 0or 1
        val_blur_sigma = 0
        if 'butter' in name_of_file:
            train_band = name_of_file.split('train-')[1].split('_test')[0]
            test_band = name_of_file.split('test-')[1]
            print('Train Band:', train_band)
            print('Test Band:', test_band)
            val_accuracy_csv = val_accuracy_csv.append({'model_pth':args.resume, 'epoch':args.start_epoch,'top1acc':f'{top1.avg:.3f}','loss':f'{losses.avg:.4e}', 'top5acc':f'{top5.avg:.3f}', 'train_band':train_band, 'test_band':test_band}, ignore_index=True)
        else:
            val_accuracy_csv = val_accuracy_csv.append({'model_pth':args.resume, 'epoch':args.start_epoch,'top1acc':f'{top1.avg:.3f}','loss':f'{losses.avg:.4e}', 'top5acc':f'{top5.avg:.3f}', 'val_images_blurred':test_images_blurred, 'val_blur_sigma':val_blur_sigma}, ignore_index=True)
        val_accuracy_csv.to_csv(output_accuracy_csv)

        # Top1Acc = []
        # Top5Acc = []

        # f = open(output_accuracy, "a")
        # f.write('Model Path: {0}\t'
        #         'Acc@1 {top1.avg:.3f}\t'
        #         # 'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
        #         'Acc@5 {top5.avg:.3f}\t'
        #         '\n'.format(
        #             args.resume, top1=top1, top5=top5))  
        # f.close

    return top1.avg

# REVISED VERSION
def shape_bias(val_loader, model, criterion, args):
    print(f"Calculating shape bias for {args.save_accuracy_file}...")

    model.eval()

    counter = 0
    tex_counter = 0
    shape_counter = 0
    other_counter = 0
    equal_counter = 0

    categories_long = ['airplane', 'bicycle', 'boat', 'car', 'chair', 'dog', 'keyboard',
                       'oven', 'bear', 'bird', 'bottle', 'cat', 'clock', 'elephant', 'knife', 'truck']
    categories_short = ['air', 'bic', 'boa', 'car', 'cha', 'dog', 'key',
                        'ove', 'bea', 'bir', 'bot', 'cat', 'clo', 'ele', 'kni', 'tru']

    categories_tex = [0] * len(categories_long)
    categories_shape = [0] * len(categories_long)
    categories_other = [0] * len(categories_long)

    # Set up path
    save_shape_bias_df = os.path.join(
        '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/exp1_blurry_vision/exp1_model_shape_bias_train_and_test_blur/shape_bias_df',
        f'{args.save_accuracy_file}.csv'
    )
    records = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            softmax_output = torch.nn.functional.softmax(output[0], dim=0)
            softmax_output_numpy = softmax_output.cpu().detach().numpy()

            # Mapping from probabilities to decision
            mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
            decision_from_16_classes = mapping.probabilities_to_decision(softmax_output_numpy)

            sample_fname, _ = val_loader.dataset.samples[i]
            short_sample_fname = os.path.basename(sample_fname)

            shape = short_sample_fname.split("-", 1)[0][:3]
            tex = short_sample_fname.split("-", 1)[1][:3]
            decision = decision_from_16_classes[:3]

            counter += 1

            # Find the full category name
            category = next((cat for short, cat in zip(categories_short, categories_long) if short == shape), "")

            # Record entry
            records.append({
                'subj': str(args.save_accuracy_file), # Model name
                # 'session': 1,
                'trial': counter,
                # 'rt': 'NaN',
                'object_response': decision_from_16_classes,
                'category': category,
                'condition': 0,
                'imagename': short_sample_fname
            })

            if shape == tex:
                equal_counter += 1
            elif shape == decision:
                shape_counter += 1
            elif tex == decision:
                tex_counter += 1
            else:
                other_counter += 1

            for idx, short in enumerate(categories_short):
                if shape == short:
                    if shape == tex:
                        pass  # equal trials
                    elif shape == decision:
                        categories_shape[idx] += 1
                    elif tex == decision:
                        categories_tex[idx] += 1
                    else:
                        categories_other[idx] += 1

        # Create DataFrame and Save
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(save_shape_bias_df), exist_ok=True)
        df.to_csv(save_shape_bias_df, index=False)

        # Shape Bias Summary
        shape_bias_total = shape_counter / (shape_counter + tex_counter) if (shape_counter + tex_counter) > 0 else float('nan')

        per_category_records = []
        for i in range(len(categories_short)):
            total = categories_shape[i] + categories_tex[i]
            shape_bias_cat = categories_shape[i] / total if total > 0 else float('nan')
            per_category_records.append({
                'Category': categories_long[i],
                'Shape_Choices': categories_shape[i],
                'Texture_Choices': categories_tex[i],
                'Other_Choices': categories_other[i],
                'Shape_Bias': shape_bias_cat
            })

        # Add total
        per_category_records.insert(0, {
            'Category': 'Total',
            'Shape_Choices': shape_counter,
            'Texture_Choices': tex_counter,
            'Other_Choices': other_counter,
            'Shape_Bias': shape_bias_total
        })

        save_shape_bias_summary = os.path.join(
            '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/exp1_blurry_vision/exp1_model_shape_bias_train_and_test_blur/shape_bias',
            f'shape_bias_{args.save_accuracy_file}_{args.test_result_stimuli_name}.csv'
        )
        df_shape_bias = pd.DataFrame(per_category_records)
        os.makedirs(os.path.dirname(save_shape_bias_summary), exist_ok=True)
        df_shape_bias.to_csv(save_shape_bias_summary, index=False)

    print(f"Overall Shape Bias: {shape_bias_total:.4f}")


    return



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



if __name__ == '__main__':
    main()



