from __future__ import print_function
import argparse
from math import log10
import sys
sys.path.append('core')
import shutil
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from sepflow import SepFlow
import evaluate
import datasets
from torch.utils.tensorboard import SummaryWriter
from utils.utils import InputPadder, forward_interpolate

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# Training settings
parser = argparse.ArgumentParser(description='PyTorch SepFlow Example')
parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--weights', type=str, default='', help="weights from saved model")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2048, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--manual_seed', type=int, default=1234, help='random seed to use. Default=123')
parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
parser.add_argument('--data_path', type=str, default='/export/work/feihu/flow/SceneFlow/', help="data root")
parser.add_argument('--save_path', type=str, default='./checkpoints/', help="location to save models")
parser.add_argument('--gpu',  default='0,1,2,3,4,5,6,7', type=str, help="gpu idxs")
parser.add_argument('--workers', type=int, default=16, help="workers")
parser.add_argument('--world_size', type=int, default=1, help="world_size")
parser.add_argument('--rank', type=int, default=0, help="rank")
parser.add_argument('--dist_backend', type=str, default="nccl", help="dist_backend")
parser.add_argument('--dist_url', type=str, default="tcp://127.0.0.1:6789", help="dist_url")
parser.add_argument('--distributed', type=int, default=0, help="distribute")
parser.add_argument('--sync_bn', type=int, default=0, help="sync bn")
parser.add_argument('--multiprocessing_distributed', type=int, default=0, help="multiprocess")
parser.add_argument('--freeze_bn', type=int, default=0, help="freeze bn")
parser.add_argument('--start_epoch', type=int, default=0, help="start epoch")
parser.add_argument('--stage', type=str, default='chairs', help="training stage: 1) things 2) chairs 3) kitti 4) mixed.")
parser.add_argument('--validation', type=str, nargs='+')
parser.add_argument('--num_steps', type=int, default=100000)
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--iters', type=int, default=12)
parser.add_argument('--wdecay', type=float, default=.00005)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
parser.add_argument('--add_noise', action='store_true')
parser.add_argument('--small', action='store_true', help='use small model')
#parser.add_argument('--smoothl1', action='store_true', help='use smooth l1 loss')

MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 2500

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions

    Args:
        flow_preds (list): list of flow predictions of shape (batch, 2, HT, WD)
        flow_gt (torch.Tensor): ground truth flow image of shape (batch, 2, HT, WD)
        valid (torch.Tensor):  whether flow_gt is valid at each pixel of shape (batch, HT, WD)
        gamma (float, optional): weight decay factor for flow refinement. Defaults to 0.8.
        max_flow (float, optional): threshold value for maximum flow value. Defaults to MAX_FLOW.

    Returns:
        Tuple[torch.Tensor,dict]: scalar tensor loss, flow quality metrics
    """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # set pixels where flow magnitude is too large as invalid
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    # in raft: weights[i] = gamma**(n_predictions-i), where i is the iteration of refinement
    # in this case: iteration 0,1 and 2 belong to inital flow regression
    #               refinement iterations start at 3
    #                       regression      refinement      
    # weights for n_img_refinement = 4:
    #   [0.1, 0.3, 0.5, 0.6024, 0.7304, 0.8904, 1.0903999999999998]
    weights = [0.1, 0.3, 0.5]
    base = weights[2] - gamma ** (n_predictions - 3)
    for i in range(n_predictions - 3):
        weights.append( base + gamma**(n_predictions - i - 4) )

    # loss based on weights array
    for i in range(n_predictions):
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += weights[i] * (valid[:, None] * i_loss).mean()

    # end point error calculation for the final flow output
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    loss_value = flow_loss.detach()
    rate0 = (epe > 1).float().mean()
    rate1 = (epe > 3).float().mean()
    error3 = epe.mean()
    
    # end point error for calculation for initial flow regression images
    epe = torch.sum((flow_preds[1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    error1 = epe.mean()
    epe = torch.sum((flow_preds[0] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    error0 = epe.mean()
    epe = torch.sum((flow_preds[2] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    error2 = epe.mean()
    
    if args.multiprocessing_distributed:
        count = flow_gt.new_tensor([1], dtype=torch.long)
        dist.all_reduce(loss_value), dist.all_reduce(error3), dist.all_reduce(error0), dist.all_reduce(error1), dist.all_reduce(error2), dist.all_reduce(count)
        dist.all_reduce(rate0), dist.all_reduce(rate1)
        n = count.item()
        loss_value, error0, error1, error2, error3 = loss_value / n, error0 / n, error1 / n, error2 / n, error3 / n
        rate1, rate0 = rate1 / n, rate0 / n

    metrics = {
        'epe0': error0.item(),
        'epe1': error1.item(),
        'epe2': error2.item(),
        'epe3': error3.item(),
        '1px': rate0.item(),
        '3px': rate1.item(),
        'loss': loss_value.item()
    }
    return flow_loss, metrics

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def main():
    
    # parse command line arguments
    args = parser.parse_args()

    # add gpu indices as environement variable
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # create list of gpu indices
    args.gpu = (args.gpu).split(',')
    
    # enable benchmarking to auto-tune computation for hardware
    # initially more overhead, can speed up computation in the long run
    torch.backends.cudnn.benchmark = True
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu.split(','))
    #args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # set seeds for random generators
    if args.manual_seed is not None:
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = True
        cudnn.deterministic = True

    # set gpus per node to number of gpus
    args.ngpus_per_node = len(args.gpu)

    # if there is one gpu, then no distributed training
    if len(args.gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        main_worker(args.gpu, args.ngpus_per_node, args)
    
    # if there are multiple gpus, create one process for each one
    # probably all processes run on the same node, since dist_url = localhost
    else:
        # TODO: assert this False
        args.sync_bn = True
        args.distributed = True
        args.multiprocessing_distributed = True
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        
        # world size goes from number of nodes to total number of gpus
        args.world_size = args.ngpus_per_node * args.world_size
        
        # spawn new process for each gpu
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler

    Args:
        args (argparse.Namespace): model parameters
        model (torch.nn.Module): model to be trained

    Returns:
        Tuple[torch.optim.AdamW, torch.optim.lr_scheduler.OneCycleLR]: create optimizer
            and learning rate scheduler for the model
    """
    modules_ori = [model.cnet, model.fnet, model.update_block, model.guidance]
    modules_new = [model.cost_agg1, model.cost_agg2]
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.lr * 2.5))
    #optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    optimizer = optim.AdamW(params_list, lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)


    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def main_process():
    """ check if the current process is the main process

    Returns:
        bool: true, if and only if the current process is the main one
    """
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def main_worker(gpu, ngpus_per_node, argss):
    """ main function executed in each process

    Args:
        gpu (string): gpu index
        ngpus_per_node (int): number of gpus per node
        argss (argparse.Namespace): arguments passed to the worker
    """

    # set process-wide arguments to the ones passed on to it
    global args
    args = argss
    
    # register current process with process group
    if args.distributed:
        # TODO: add assert to check that this is not executed
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # create Separable Flow module and get the optimizer
    model = SepFlow(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    # synchronize batchnorm between processes
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # if there are multiple gpus, DistributedDataParallel is usually faster for any number of nodes
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batchSize = int(args.batchSize / args.ngpus_per_node)
        args.testBatchSize = int(args.testBatchSize / args.ngpus_per_node)
        args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    
    # if there is only one gpu, DataParallel is simpler to use
    else:
        model = torch.nn.DataParallel(model).cuda()

    #scheduler = None

    # create logger
    logger = Logger(model, scheduler)

    # load weights at the start of training
    if args.weights:
        if os.path.isfile(args.weights):
            checkpoint = torch.load(args.weights, map_location=lambda storage, loc: storage.cuda())
            msg=model.load_state_dict(checkpoint['state_dict'], strict=False)
            if main_process():
                print("=> loaded checkpoint '{}'".format(args.weights))
                print(msg)
                sys.stdout.flush()
        else:
            if main_process():
                print("=> no checkpoint found at '{}'".format(args.weights))
    
    # load previous weights, but also optimizer and scheduler state 
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            msg=model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            args.start_epoch = checkpoint['epoch'] + 1
            if main_process():
                print("=> resume checkpoint '{}'".format(args.resume))
                print(msg)
                sys.stdout.flush()
        else:
            if main_process():
                print("=> no checkpoint found at '{}'".format(args.resume))

    # create and load datasets for validation
    train_set = datasets.fetch_dataloader(args)
    val_set = datasets.KITTI(split='training')
    val_set3 = datasets.FlyingChairs(split='validation')
    val_set2_2 = datasets.MpiSintel(split='training', dstype='final')
    val_set2_1 = datasets.MpiSintel(split='training', dstype='clean')
    
    sys.stdout.flush()
    
    # create DistributedSampler for datasets to load only relevant subset of dataset in each process
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        val_sampler2_2 = torch.utils.data.distributed.DistributedSampler(val_set2_2)
        val_sampler2_1 = torch.utils.data.distributed.DistributedSampler(val_set2_1)
        val_sampler3 = torch.utils.data.distributed.DistributedSampler(val_set3)
    else:
        train_sampler = None
        val_sampler = None
        val_sampler2_1 = None
        val_sampler2_2 = None
        val_sampler3 = None

    # create dataloader from Datset/DistributedSampler, for automated batching/shuffling
    # also responsible for preloading batches to RAM, before sending them to the gpu
    training_data_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batchSize, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=args.testBatchSize, shuffle=False, num_workers=args.workers//2, pin_memory=True, sampler=val_sampler)
    val_data_loader2_2 = torch.utils.data.DataLoader(val_set2_2, batch_size=args.testBatchSize, shuffle=False, num_workers=args.workers//2, pin_memory=True, sampler=val_sampler2_2)
    val_data_loader2_1 = torch.utils.data.DataLoader(val_set2_1, batch_size=args.testBatchSize, shuffle=False, num_workers=args.workers//2, pin_memory=True, sampler=val_sampler2_1)
    val_data_loader3 = torch.utils.data.DataLoader(val_set3, batch_size=args.testBatchSize, shuffle=False, num_workers=args.workers//2, pin_memory=True, sampler=val_sampler3)

    error = 100
    args.nEpochs = args.num_steps // len(training_data_loader) + 1

    # epoch-wise training (other than raft)
    for epoch in range(args.start_epoch, args.nEpochs):
        
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train model for one epoch, e.g. numSteps / nEpochs
        train(training_data_loader, model, optimizer, scheduler, logger, epoch)
        
        # save check point only after the last three epochs
        if main_process(): #and epoch > args.nEpochs - 3:
            save_checkpoint(args.save_path, epoch,{
                    'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer' : optimizer.state_dict(),
                     'scheduler' : scheduler.state_dict(),
                 }, False)
        
        # for chairs, validate on the chairs validation set
        if args.stage == 'chairs':
            loss = val(val_data_loader3, model, split='chairs')

        # for sintel and things, validate on the sintel clean and final dataset 
        # and kitti training dataset
        elif args.stage == 'sintel' or args.stage == 'things':
            loss_tmp = val(val_data_loader2_1, model, split='sintel', iters=32)
            loss_tmp = val(val_data_loader2_2, model, split='sintel', iters=32)
            
            # results in error
            # loss_tmp = val(val_data_loader, model, split='kitti')
        
        # for kitti, valdiate on the kitti training dataset
        elif args.stage == 'kitti':
            
            # results in error
            # loss_tmp = val(val_data_loader, model, split='kitti')

            pass

    # if the current process is the main process, then save the model one final time after training
    if main_process():
        save_checkpoint(args.save_path, args.nEpochs,{
                'state_dict': model.state_dict()
            }, True)


def train(training_data_loader, model, optimizer, scheduler, logger, epoch):
    """ train the model for one epoch

    Args:
        training_data_loader (torch.utils.data.DataLoader): data loader for the epoch
        model (torch.nn.Module): torch model
        optimizer (torch.optim.Optimizer): optimizer used in training
        scheduler (torch.optim.lr_scheduler.some_scheduler): learning rate scheduler
        logger (Logger): logger for training stats
        epoch (int): current epoch number
    """
    
    # number of successful iterations/batches
    valid_iteration = 0
    
    # put model in training mode
    model.train()

    # freeze parameters of the batch norm
    if args.freeze_bn:
        model.module.freeze_bn()
        if main_process():
            print("Epoch " + str(epoch) + ": freezing bn...")
            sys.stdout.flush()
    
    # iterate over all batches in the dataset
    for iteration, batch in enumerate(training_data_loader):
        
        # probably deprecated use of "Variable"
        # Variable was at some point required to enable the forward pass for input tensors to the model
        input1, input2, target, valid = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False)
        
        # load inputs, target and valid to gpu asynchronously
        input1 = input1.cuda(non_blocking=True)
        input2 = input2.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        valid = valid.cuda(non_blocking=True)

        # if valid has more dimensions than (batch, HT, WD), then remove dimensions of size 1
        if len(valid.shape) > 3:
            valid = valid.squeeze(1)

        # check if there is at least one pixel with valid flow in the batch
        # if there is None, there is nothing to do
        if valid.sum() > 0:

            # set all gradients to zero
            optimizer.zero_grad()
            
            # additive noise with random magnitude
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                input1 = (input1 + stdv * torch.randn(*input1.shape).cuda()).clamp(0.0, 255.0)
                input2 = (input2 + stdv * torch.randn(*input2.shape).cuda()).clamp(0.0, 255.0)

            # get predictions from model
            flow_predictions = model(input1, input2, iters=args.iters)            
            
            # calculate loss
            loss, metrics = sequence_loss(flow_predictions, target, valid)

            # backpropagation of loss gradient
            loss.backward()

            # update parameters
            optimizer.step()

            # update learning rate
            scheduler.step()
            
            # after each batch, change learning rate
            # this is done to set the learning rate of the cost_agg1 and cost_agg2 module
            # 2.5 times higher than the rest of the modules
            adjust_learning_rate(optimizer, scheduler)
            
            # if learning rate becomes too small, stop epoch early
            if  scheduler.get_last_lr()[0] < 0.0000002:
                return

            # iteration counter only increased if iteration was successful
            valid_iteration += 1

            # logging and saving only in main process
            if main_process():
                logger.push(metrics)

                # every 10000 valid iterations save checkpoint (might not even occurr in an epoch)
                if valid_iteration % 10000 == 0:

                    save_checkpoint(args.save_path, epoch,{
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'scheduler' : scheduler.state_dict(),
                        }, False)

            sys.stdout.flush()

def val(testing_data_loader, model, split='sintel', iters=24):
    
    epoch_error = 0
    epoch_error_rate0 = 0
    epoch_error_rate1 = 0
    valid_iteration = 0
    
    model.eval()
    
    # error in the line below
    # for testing_data_loader of kitti: RuntimeError: Trying to resize storage that is not resizable
    for iteration, batch in enumerate(testing_data_loader):
        
        # create tensors/variables for inputs
        input1, input2, target, valid = Variable(batch[0],requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False)
        
        # put input image tensors on gpu
        input1 = input1.cuda(non_blocking=True)
        input2 = input2.cuda(non_blocking=True)
        
        # pad inputs to make them divisible by 8
        padder = InputPadder(input1.shape, mode=split)
        input1, input2 = padder.pad(input1, input2)
        
        # put target flow and flow valid flags to gpu
        target = target.cuda(non_blocking=True)
        valid = valid.cuda(non_blocking=True)

        # calculate magnitude of target flow vectors
        mag = torch.sum(target**2, dim=1, keepdim=False).sqrt()
        
        # remove dimensions of size 1
        # i.e. (batch, 1, HT, WD) -> (batch, HT, WD)
        if len(valid.shape) > 3:
            valid = valid.squeeze(1)

        # convert valid floating point 0-1-tensor to bool tensor 
        valid = (valid >= 0.001) #& (mag < MAX_FLOW)
        
        # only evaluate if at least one pixel has valid flow ground truth
        if valid.sum()>0:
            
            # evaluate the model on the inputs, caluclate epe
            with torch.no_grad():
                
                # get the flow
                _, flow = model(input1,input2, iters=iters)
                
                # remove padding
                flow = padder.unpad(flow)
                
                # calculate endpoint error
                epe = torch.sum((flow - target)**2, dim=1).sqrt()
                
                # remove all invalid pixel epes from the array
                epe = epe.view(-1)[valid.view(-1)]
                
                # calculate percentage of all pixels with epe>1
                rate0 = (epe > 1).float().mean()
                
                # calculate percentage of all pixels with epe>3
                if split == 'kitti':
                    # percentage of pixels with epe>3 and epe/target_flow_magnitude > 0.05
                    # -> only count pixels where epe is at least 5% as large as the target flow magnitude 
                    rate1 = ((epe > 3.0) & ((epe/mag.view(-1)[valid.view(-1)]) > 0.05)).float().mean()
                else:
                    rate1 = (epe > 3.0).float().mean()
                
                # get the average per-pixel epe
                error = epe.mean()

                # count iteration as successful
                valid_iteration += 1
            
            # not executed
            # calculates the epoch metrics, no metrics are calculated if not multiprocessing
            if args.multiprocessing_distributed:
                
                # TODO: add assert for not executed
                
                # add 1-tensor for worker counting
                count = target.new_tensor([1], dtype=torch.long)
                
                # collect metric values from workers and sum them
                dist.all_reduce(error)
                dist.all_reduce(rate0)
                dist.all_reduce(rate1)
                
                # calculate number of workers executing this
                dist.all_reduce(count)
                n = count.item()
                
                # divide by number of workers
                error /= n
                rate0 /= n
                rate1 /= n

                # add to total epoch error
                epoch_error += error.item()
                epoch_error_rate0 += rate0.item()
                epoch_error_rate1 += rate1.item()
            else:

                # much simpler for non-distributed version
                epoch_error += error.item()
                epoch_error_rate0 += rate0.item()
                epoch_error_rate1 += rate1.item()

            # valid iteration never becomes 1000, thus the message below is never printed
            if main_process() and (valid_iteration % 1000 == 0):
                print("===> Test({}/{}): Error: ({:.4f} {:.4f} {:.4f})".format(iteration, len(testing_data_loader), error.item(), rate0.item(), rate1.item()))
            
            sys.stdout.flush()

    # print epoch validation info, always zero without multiprocessing because of the above
    if main_process():
        print("===> Test: Avg. Error: ({:.4f} {:.4f} {:.4f})".format(epoch_error/valid_iteration, epoch_error_rate0/valid_iteration, epoch_error_rate1/valid_iteration))

    return epoch_error/valid_iteration

def save_checkpoint(save_path, epoch,state, is_best):
    """ simple method to store checkpoint including current epoch, weights, optimizer and lr_scheduler

    Args:
        save_path (str): (relative) path to store checkpoint
        epoch (int): current epoch
        state (dict): dictionary containing weights, optimizer and lr_scheduler
        is_best (bool): whether this model achieves the new best performance
    """
    filename = save_path + "_epoch_{}.pth".format(epoch)
    
    # save best model without the epoch tag
    if is_best:
        filename = save_path + ".pth"
    
    torch.save(state, filename)
    
    print("Checkpoint saved to {}".format(filename))


def adjust_learning_rate(optimizer, scheduler):
    """ adjust optimizer learning rates individually for the parameter groups

    Args:
        optimizer (torch.optim.Optimizer): parameter optimizer
        scheduler (torch.optim.lr_scheduler.OneCycleLR): learning rate updater
    """

    # use the only the first component of the latest learning rate
    # the array returned by get_last_lr has six identical learning rates for the parameter groups
    # i.e.: scheduler.get_last_lr() = np.ones((len(param_groups)))*lr
    # where the six groups are: 
    # [model.cnet, model.fnet, model.update_block, model.guidance, model.cost_agg1, model.cost_agg2]
    lr = scheduler.get_last_lr()[0]
    nums = len(optimizer.param_groups)

    # for first nums-2 groups, set learning rate to lr
    # learninig rate does not change: lr -> lr
    # they are: [model.cnet, model.fnet, model.update_block, model.guidance]
    for index in range(0, nums-2):
        optimizer.param_groups[index]['lr'] = lr

    # for last two groups, set learning rate to lr*2.5
    # learning rate changes: lr -> 2.5*lr
    # they are: [model.cost_agg1, model.cost_agg2]
    # this means that the learning rate for 1d cost aggregation 
    # is chosen as 2.5 times higher than for the rest of the parameters
    for index in range(nums-2, nums):
        optimizer.param_groups[index]['lr'] = lr * 2.5

if __name__ == '__main__':
    main()
