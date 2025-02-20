# Filename and Gauss parsers are working and may be included directly into bash script

# Below No longer Required
# NB change filename in 3 places, adjust gaussian blur as required
# TESTING whether cpkt_name PARSER HAS FIXED THIS
# - IF SO ONLY 1 THING TO CHANGE - GAUSSIAN BLUR!!! - COULD MAKE A PARSER WOLD NEED A ZERO OPTION - LOOP OR 0 IN FUNCTION?
# Currently - Gaussian is set to 6
#Add these changes ie gaussia to main_general once save cpkts is working

import argparse
import os
import random
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

import pandas as pd
import datetime


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

print(f'Model Names: {model_names}')

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
parser.add_argument('--cpkt_name', default=None, type=str, metavar='CPKT_name',
                    help='name checkpoints')  #Name checkpoints a from bash script without adapting python script

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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
parser.add_argument('--gauss', default=0, type=float, metavar='N',
                    help='amount of gaussian blur to apply to the train_loader') #Added 
parser.add_argument('--kernel', default=9, type=int, metavar='N',
                    help='size of kernel for gaussian blur applied to train_loader')
parser.add_argument('--save_batch_freq', type=int, default=1001, help='save frequency within an epoch') 
parser.add_argument('--path_acc', type=str, default=None, help='path to save accuracy')
parser.add_argument('--iteration', type=int, default=None, help='Replication of Training')
parser.add_argument('--batches_to_save', type=str, default=None, help='path to csv with batch IDs to save in n_batches_rounded column')
parser.add_argument('--lr_freq', type=int, default=30, help='frequency to change the learning rate in epochs') 


opt = parser.parse_args() #Added from train_CMC.py to facilitate saving ckpts
global args #Attempt to make global to facilitate custimization of filename! 
args = parser.parse_args() #Added from Main - remove all opt eventually 



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
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'): #Revert to default 
        # if ((args.arch.startswith('alexnet')) or (args.arch.startswith('vgg'))):
        # # if args.arch.startswith('alexnet'):
        #     print('RUNNING')
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            print('NOT RUNNING!!!')
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
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
#     valdir = os.path.join(args.data, 'val') # Need to change Val to val in folders as this is the directory name on server
    valdir = os.path.join(args.data, 'val_in_folders')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    print(f'Confirm Parameters are set correctly:\n\targs.gauss: {args.gauss}, args.kernel: {args.kernel}')
   

    print(f'valdir: {valdir}, traindir: {traindir}')

 
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
 


    #model_names was causing errors - removed to simplify - using DeepCluster vesion instead
#     #Model Folder to facilitate saving of checkpoints - from train_CMC.py line 113-116
#     opt.model_folder = os.path.join(opt.model_path, opt.model_names) #Changed model_name to model_names 
#     if not os.path.isdir(opt.model_folder):
#         os.makedirs(opt.model_folder)
        
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    n_batches_per_epoch = len(train_loader)
    print(f'Length of train_loader: {n_batches_per_epoch}')

    # Load path to save accuracy
    # Attempted to modify for all band!
    # band = args.cpkt_name.replace('butter_','').split('bp_')[1].split('_')[0]
    # print(f'Band is {band}')

    if args.batches_to_save:
        batches_df = pd.read_csv(args.batches_to_save, index_col=0)
        batches_df['n_batches_rounded'] = batches_df['n_batches_rounded'] - 1   
        # batch_ids_to_save = batches_df['n_batches_rounded'].unique() 
        # if alexnet
        # 
        if args.batch_size == 32:
            batches_df['n_batches_rounded'] = batches_df['n_batches_rounded'] * 8
    else:
        batch_ids_to_save = None

    # print(f'Batch IDs to Save: {batch_ids_to_save}')
    

    band = args.cpkt_name.split('butter_')[1].split('_')[0]

    # output_accuracy_csv = os.path.join(args.path_acc, f'supervised_resnet50_bp_butter_train-{band}_test-{band}_iter-{args.iteration}_accuracy.csv')
    output_accuracy_csv = os.path.join(args.path_acc, f'supervised_{args.arch}_bp_butter_train-{band}_test-{band}_log_iter-{args.iteration}_accuracy.csv')

    if os.path.exists(output_accuracy_csv):
        val_accuracy_csv = pd.read_csv(output_accuracy_csv, index_col=0)
    else:
        val_accuracy_csv = pd.DataFrame(columns=['model_pth','epoch', 'batch', 'top1acc', 'top5acc', 'loss', 'train_band', 'test_band', 'timestamp']) #add column for epoch (perhaps instead of path?)
    val_accuracy_csv.to_csv(output_accuracy_csv)

    # if not resuming, start from scratch
    if args.resume == '':
        # Save checkpoint for untrained model
        ckpt_suffix = 'untrained' + '_log' # Added _log to distinguish from other analysis
        ckpt_file_name = save_checkpoint(model, 0, optimizer, args.cpkt_name, ckpt_suffix=ckpt_suffix, iteration=args.iteration)
        
        val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion, args)

        save_evaluation_results(output_accuracy_csv, ckpt_file_name, 0, None, val_acc1, val_acc5, val_loss, band)

   

    for epoch in range(args.start_epoch, args.epochs):
        print(f'Epoch: {epoch}')
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        epoch_rows_df = batches_df[batches_df['corresponding_epoch'] == epoch]
        batch_ids_to_save_for_epoch = epoch_rows_df['n_batches_rounded'].unique()

        # train for one epoch
        train(train_loader, val_loader, model, criterion, optimizer, epoch, args, output_accuracy_csv, band, batch_ids_to_save_for_epoch, n_batches_per_epoch)


        # SAVE FOR ALL EPOCHS IN THIS SCRIPT
        ckpt_suffix = 'complete' + '_log' #Added _log to distinguish from other analysis
        
        # f'checkpoint_{str(cpkt_name)}_epoch{str(epoch)}{suffix}.pth.tar'
        ckpt_file_name = save_checkpoint(model, epoch, optimizer, args.cpkt_name, ckpt_suffix=ckpt_suffix, iteration=args.iteration)

        # TO DO: save accuracy to csv!
        # evaluate on validation set
        val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion, args)
  
        save_evaluation_results(output_accuracy_csv, ckpt_file_name, epoch, n_batches_per_epoch, val_acc1, val_acc5, val_loss, band)
        # '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/bandpass_analysis_butterworth'

        
        

def train(train_loader, val_loader, model, criterion, optimizer, epoch, args, output_accuracy_csv, band=None, batch_ids_to_save=None, n_batches_per_epoch=None):
    if n_batches_per_epoch is None:
        n_batches_per_epoch = len(train_loader)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        n_batches_per_epoch,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    # Ensure not overwriting batch_ids_to_save
    # FIX
    # batch_ids_to_save_for_epoch = batch_ids_to_save + epoch*len(train_loader)

    # TO DO - Pass in Epoch specific batches

    
    


    # switch to train mode
    model.train()

    print(f'Training epoch {epoch}...')

    end = time.time()
    for i, (images, target) in enumerate(train_loader): # i is the batch number
        cum_batch_no = i + epoch * n_batches_per_epoch 
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
        # # create new variable called save_batch_freq
        # if ((i !=0) and (i % args.save_batch_freq == 0)):

        if cum_batch_no in batch_ids_to_save:
        # if (i + epoch * n_batches_per_epoch) in batch_ids_to_save: # Initial incorrect 
        # if (i + epoch * 5005) in batch_ids_to_save_for_epoch: #In process of fixing
        # if i % args.save_batch_freq == 0:
            print(f'{cum_batch_no} batches have been trained')
            # acc1 = validate(val_loader, model, criterion, args)
            # print(f'==> Saving Evaluation Results for batch {i}...')
            
            # ckpt_suffix = 'batch'+str(i) # 
            ckpt_suffix = 'batch'+str(cum_batch_no)+'_log' #Added _log to distinguish from other analysis
            
            ckpt_file_name = save_checkpoint(model, epoch, optimizer,  args.cpkt_name, ckpt_suffix=ckpt_suffix, iteration=args.iteration)

            # REMOVE EVALUATION FROM TRAINING LOOP TO SPEED UP TRAINING - can calculate after training if required!!! 26/8/24
            # val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion, args)

            # batch = i
            # save_evaluation_results(output_accuracy_csv, ckpt_file_name, epoch, batch, val_acc1, val_acc5, val_loss, band)

            # # Make sure model is in train mode before resuming training!!
            # # Does this fix the bug?? Fri 23-8-24
            # model.train()

            # # calculate accuracy and loss and save to csv!
        


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

    return top1.avg, top5.avg , losses.avg


def save_evaluation_results(output_accuracy_csv, ckpt_file_name, epoch, batch, acc1, acc5, loss, band):
    print(f'==> Saving Evaluation Results for epoch {epoch} batch: {batch}')
    val_accuracy_csv = pd.read_csv(output_accuracy_csv, index_col=0)
    val_accuracy_csv = val_accuracy_csv.append({'model_pth':ckpt_file_name, 'epoch':epoch,'batch':batch,'top1acc':f'{acc1:.3f}','top5acc':f'{acc5:.3f}', 'loss':f'{loss:.4e}', 'train_band':band, 'test_band':band, 'timestamp': str(datetime.datetime.now())}, ignore_index=True)
    val_accuracy_csv.to_csv(output_accuracy_csv)


# Created function 17 Aug 24
def save_checkpoint(model, epoch, optimizer, cpkt_name, ckpt_suffix=None, iteration=None): 
    print(f'==> Saving Checkpoint... Epoch: {epoch} {ckpt_suffix}')
    
    if ckpt_suffix != None:
        ckpt_suffix = f'_{ckpt_suffix}'
    if iteration != None:
        iteration = f'_iter-{iteration}'
    ckpt_file_name = f'checkpoint_{str(cpkt_name)}_epoch{str(epoch)}{ckpt_suffix}{iteration}.pth.tar'
    
    torch.save({'epoch': epoch + 1,
                'arch': args.arch, # CAUTION not currently passed in!!
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()},
                os.path.join(args.model_path, ckpt_file_name))
    print(f'==> Checkpoint saved as: {ckpt_file_name} to {args.model_path}')
    return ckpt_file_name



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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr * (0.1 ** (epoch // args.lr_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


if __name__ == '__main__':
    main()