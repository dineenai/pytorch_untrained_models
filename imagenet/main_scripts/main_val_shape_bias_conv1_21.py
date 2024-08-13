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
# import torchvision.models as models
import resnet_conv1_21 as models
import ntpath #Split path: https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
from pytorch_image_folder_with_file_paths import ImageFolderWithPaths #Print Image Name https://stackoverflow.com/questions/56962318/printing-image-paths-from-the-dataloader-in-pytorch
import probabilities_to_decision
from probabilities_to_decision import ImageNetProbabilitiesTo16ClassesMapping
import helper

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

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
#     valdir = os.path.join(args.data, 'val') # Need to change Val to val in folders as this is the directory name on server
    # valdir = os.path.join(args.data, 'val')
    # valdir = os.path.join(args.data, 'val') #Change to val_in_folders for imagenet, ie DIR='/data/ILSVRC2012/'  TO DO add this to loop 
    valdir = os.path.join(args.data, 'style-transfer-preprocessed-512') 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    #model_names was causing errors - removed to simplify - using DeepCluster vesion instead
#     #Model Folder to facilitate saving of checkpoints - from train_CMC.py line 113-116
#     opt.model_folder = os.path.join(opt.model_path, opt.model_names) #Changed model_name to model_names 
#     if not os.path.isdir(opt.model_folder):
#         os.makedirs(opt.model_folder)
        
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # folder_dataset_val = ImageFolderWithPaths(root=Config.valdir)
    # Note that this is not actually connected to the val_loader!
    folder_dataset_val = ImageFolderWithPaths(valdir) #works

        
        # print(paths)
    #Still need to encorporate the above into loader      
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # test_dataloader = torch.utils.data.DataLoader(folder_dataset_val,num_workers=6,batch_size=1,shuffle=True)    

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
   
    

#         # save model - from train_CMC.py
#         if epoch % args.save_freq == 0:
#             print('==> Saving...')
#             state = {
#                 'opt': args,
#                 'model': model.state_dict(),
#                 'contrast': contrast.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'epoch': epoch,
#             }
#             if args.amp:
#                 state['amp'] = amp.state_dict()
#             save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
#             torch.save(state, save_file)
#             # help release GPU memory
#             del state

        #Nb adapten and added to save_checlpoint definition ####       
        # if epoch % args.save_freq == 0:
        #     print('==> Saving...')
        # # save running checkpoint From Anna T DeepCluster https://github.com/AnnaTruzzi/deepcluster/blob/backtostart/main.py
        #     torch.save({'epoch': epoch + 1,
        #                 'arch': args.arch,
        #                 'state_dict': model.state_dict(),
        #                 'optimizer' : optimizer.state_dict()},
        #              #  os.path.join(args.exp, 'checkpoint_dc'+str(args.instantiation)+'_epoch'+str(epoch)+'.pth.tar'))
        #                os.path.join(args.model_path, 'checkpoint_unsupervised_resnet50'+'_epoch'+str(epoch)+'.pth.tar'))


# def train(train_loader, model, criterion, optimizer, epoch, args):
#     batch_time = AverageMeter('Time', ':6.3f')
#     data_time = AverageMeter('Data', ':6.3f')
#     losses = AverageMeter('Loss', ':.4e')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
#     progress = ProgressMeter(
#         len(train_loader),
#         [batch_time, data_time, losses, top1, top5],
#         prefix="Epoch: [{}]".format(epoch))

#     # switch to train mode
#     model.train()

#     end = time.time()
#     for i, (images, target) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)

#         if args.gpu is not None:
#             images = images.cuda(args.gpu, non_blocking=True)
#         if torch.cuda.is_available():
#             target = target.cuda(args.gpu, non_blocking=True)

#         # compute output
#         output = model(images)
#         loss = criterion(output, target)

#         # measure accuracy and record loss
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         losses.update(loss.item(), images.size(0))
#         top1.update(acc1[0], images.size(0))
#         top5.update(acc5[0], images.size(0))

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % args.print_freq == 0:
#             progress.display(i)


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

    counter = 0
    tex_counter = 0
    shape_counter = 0
    other_counter = 0
    equal_counter = 0

    categories_long = ['airplane', 'bicycle', 'boat', 'car', 'chair', 'dog', 'keyboard',
                        'oven', 'bear', 'bird', 'bottle', 'cat', 'clock', 'elephant', 'knife', 'truck']

    categories_short = ['air', 'bic', 'boa', 'car', 'cha', 'dog', 'key',
                        'ove', 'bea', 'bir', 'bot', 'cat', 'clo', 'ele', 'kni', 'tru']

    # listofzeros = [0] * n
    categories_tex = [0] * len(categories_long)
    categories_shape = [0] * len(categories_long)
    categories_other = [0] * len(categories_long)
    categories_shape_bias = [0] * len(categories_long)


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            # softmax_output = SomeCNN(input_image)
          
            # Get output type
            # print("output: ")
            # print(type(output)) #<class 'torch.Tensor'>

            # softmax_output_numpy = SomeConversionToNumpy(output)

            # convert to numpy
            # https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array
            # np_arr = torch_tensor.cpu().detach().numpy()



            # NOT PROBABILITIES BUT OTHERWISE WRKING!!!
            # softmax_output_numpy = output.cpu().detach().numpy()

            # probabilities = torch.nn.functional.softmax(output[0], dim=0)
            softmax_output = torch.nn.functional.softmax(output[0], dim=0)
            softmax_output_numpy = softmax_output.cpu().detach().numpy()



            # print("Numpy?: ")
            # print(type(softmax_output_numpy)) #<class 'numpy.ndarray'>

            # softmax_output_numpy = SomeConversionToNumpy(softmax_output) # replace with conversion
            
            # From Below REM: need softmax output!!
            # probabilities = torch.nn.functional.softmax(output[0], dim=0)

            
            # create mapping
            mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
    
            # obtain decision 
            decision_from_16_classes = mapping.probabilities_to_decision(softmax_output_numpy)

            # # https://stackoverflow.com/questions/56699048/how-to-get-the-filename-of-a-sample-from-a-dataloader
            sample_fname, _ = val_loader.dataset.samples[i]
            print(sample_fname)
            short_sample_fname = ntpath.basename(sample_fname) 

            print(short_sample_fname)
            # print(ntpath.basename(sample_fname))
            print(decision_from_16_classes)

            short_decision = decision_from_16_classes[ 0: 3]
            print(short_decision)

            # decision_from_16_classes
            # short_sample_fname
            # tex = short_sample_fname[ 0: 3]
            shape = short_sample_fname.split("-",1)[0]
            short_shape = shape[ 0: 3]
            tex = short_sample_fname.split("-",1)[1]
            short_tex = tex[ 0: 3]
            # print("shape")
            # print(short_shape)
            # print("tex")
            # print(short_tex)

            counter+=1

            if (short_shape == short_tex):
                equal_counter+=1
            elif(short_shape == short_decision):
                shape_counter+=1
            elif(short_tex == short_decision):
                tex_counter+=1
            else:
                other_counter+=1
            
            # TO DO: FIX BELOW
            

            # initialise list of len(categories) 
            # Fill with counter str(category)
 
            for i in range(0, len(categories_short)):
                # if (short_decision == categories_short[i]):
                if (short_shape == categories_short[i]): #FIXED?
                    print(categories_long[i])
                    if (short_shape == short_tex):
                        print("same")
                    elif(short_shape == short_decision):
                        categories_shape[i]+=1
                    elif(short_tex == short_decision):
                        categories_tex[i]+=1
                    else:
                        categories_other[i]+=1

        # print(counter)
        # print(tex_counter)
        # print(shape_counter)
        # print(other_counter)
        print("Shape Bias: ")
        shape_bias = (shape_counter/(shape_counter + tex_counter))
        print(shape_bias)
        # print(shape_counter/(shape_counter + tex_counter)) 
        # print(categories_long) 
 
        # print("categories_shape")
        # print(categories_shape)
        # print("categories_tex")
        # print(categories_tex)

        
        for i in range(0, len(categories_short)):
            categories_shape_bias[i] = (categories_shape[i])/(categories_shape[i] + categories_tex[i])
   
        # print(categories_shape_bias) 

        save_shape_bias = '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/shape_bias/shape_bias_'+str(args.save_accuracy_file)+'_'+str(args.test_result_stimuli_name)+'.txt'
        f1 = open(save_shape_bias, 'a')
        f1.write('Category, '
                    'Shape_Choices, '
                    'Texture_Choices, '
                    'Other_Choices, '
                    'Shape_Bias, ' )

        f1.write('Total, '
                '{0}, '
                '{1}, '
                '{2}, '
                '{3}, '
                '\n'.format(
                    shape_counter, tex_counter,
                    other_counter, shape_bias))

        for i in range(0, len(categories_short)):
            f1.write('{0}, '
                '{1}, '
                '{2}, '
                '{3}, '
                '{4}, '
                '\n'.format(
                    categories_long[i], categories_shape[i], categories_tex[i],
                    categories_other[i], categories_shape_bias[i]))   
        f1.close



            # Break filename into two parts - before and after hyphen
            # if before = after exclude
            # else
            # if before = decision - shappe
            # if after = decision = texture
            # counters.......
            # add 
            # shape / total = shape bias

            # 451-462 - commented out!
#             loss = criterion(output, target)
#             # print("Output: ")
#             # print(output) #a tensor

#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(output, target, topk=(1, 5))
#             losses.update(loss.item(), images.size(0))
#             top1.update(acc1[0], images.size(0))
#             top5.update(acc5[0], images.size(0))

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             # # https://stackoverflow.com/questions/56699048/how-to-get-the-filename-of-a-sample-from-a-dataloader
#             sample_fname, _ = val_loader.dataset.samples[i]
#             print(sample_fname)
#             short_sample_fname = ntpath.basename(sample_fname) 
#             print(short_sample_fname)

#             # # Added from CMC LinearProbing_val.py
#             save_test_result = '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/test_results/test_results_'+str(args.save_accuracy_file)+'/'+str(args.save_accuracy_file)+'_'+str(args.test_result_stimuli_name)+'.txt'
#             f1 = open(save_test_result, 'a')
#             # f1 = open('test_result.txt', 'a') #This one works - try above instead!
#             # if i % opt.print_freq == 0:
#             if i % args.print_freq == 0:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                       'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                        i, len(val_loader), batch_time=batch_time, loss=losses,
#                        top1=top1, top5=top5))

#                 # NB this works for test data set BUT will need to use folder title and link the offset actual label####
#                 #Make a loop, perhapse use a parser specifying imagenet!

#                 # Added from: https://pytorch.org/hub/pytorch_vision_resnet/
#                 # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
 
#                 # DO I NEED THIS? - YES - USED BELOW
#                 # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
#                 probabilities = torch.nn.functional.softmax(output[0], dim=0)
#                 # print(probabilities)

#                 # This may not be the best location! #MOVE?

#                 # Download ImageNet labels: 
#                 # removed ! and executed in command line:
#                 # !wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
                
#                 # Read the categories
#                 with open("imagenet_classes.txt", "r") as f2:
#                     categories = [s.strip() for s in f2.readlines()]
#                 # Show top categories per image
#                 top5_prob, top5_catid = torch.topk(probabilities, 5)
#                 for i in range(top5_prob.size(0)):
#                     print(categories[top5_catid[i]], top5_prob[i].item())

#                 # # Can I do the same for target as done for outout above to get the images' label - ie for imagenet validation images?

             
#                  # TO DO: PRINT ACTUAL FILE NAME NB - FIX
#                 #Added Code
#                 f1.write('Image: {0}\t'
#                         'Test: [{1}/{2}]\t'
# #                         'Output: {out}\t'
#                         'Loss: {loss.val:.4f}\t'
#                         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                         'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
#                         '\n'.format(
#                          short_sample_fname, i, len(val_loader), loss=losses,
#                          top1=top1, top5=top5))                   
                

#                 for i in range(top5_prob.size(0)):
#                     print(categories[top5_catid[i]], top5_prob[i].item())
#                     f1.write('Prediction {0}: {1}, {2}\t'
#                             '\n'.format(
#                                 i + 1, categories[top5_catid[i]], top5_prob[i].item()))

#                 f2.close
#             f1.close 

#             if i % args.print_freq == 0:
#                 progress.display(i)

#         # TODO: this should also be done with the ProgressMeter
#         print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
#               .format(top1=top1, top5=top5))
        
#         # # OUTPUT MODEL ACCURACY 
#         # # Added file to save average accuracy following validation, added parsers
#         # # save_path = '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/model_accuracy/'
#         # save_path = args.save_accuracy_path
#         # # name_of_file = raw_input("What is the name of the file: ") #Add Parser to capture network name!
#         # # name_of_file = 'testfile_model_accuracy'
#         # name_of_file = args.save_accuracy_file
#         # output_accuracy = os.path.join(save_path, name_of_file+".txt") 

        
        
#         # # f = open('model_accuracy.txt', 'a')
#         # f = open(output_accuracy, "a")
#         # f.write('Model Path: {0}\t'
#         #         'Acc@1 {top1.avg:.3f}\t'
#         #         # 'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
#         #         'Acc@5 {top5.avg:.3f}\t'
#         #         '\n'.format(
#         #             args.resume, top1=top1, top5=top5))  
#         # f.close

    return top1.avg




# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')

def save_checkpoint(state, is_best, filename):
    torch.save(state, 'checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')         
    #Added - save every 5 epochs
    if epoch % args.save_freq -1 == 0: #Does the minus 1 work?, does this require brackets?
        print('==> Saving...')
        torch.save(state, os.path.join(args.model_path, 'checkpoint_unsupervised_resnet50'+'_epoch'+str(epoch)+'.pth.tar'))


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
    lr = args.lr * (0.1 ** (epoch // 30))
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
