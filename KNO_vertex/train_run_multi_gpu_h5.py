#!/usr/bin/env python
import numpy as np
import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import DataLoader
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Data
import sys, os
import subprocess
import csv, yaml
import math
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.optim as optim
import argparse
import pandas as pd
from loss_functions import *
import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score
from tqdm import tqdm


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

sys.path.append("./module")
from model.allModel import *
from datasets.dataset_h5_v1 import *




def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', action='store', type=str)
    parser.add_argument('-o', '--output', action='store', type=str)
    parser.add_argument('--device', nargs="+", default=['0','1','2','3'])
    parser.add_argument('--rank_i', type=int, default=0)

    return parser



### world size : 사용되는 프로세스들의 갯수 = 분산 처리에서 사용되는 총 gpu개수
### rank : process(GPU)의 id
### global rank : 전체 node에서의 id
### local rank : 각 node에서의 id


def main_worker(rank,args):

    # print(args.rank,'erererer')
    local_gpu_id = init_for_distributed(rank,args)

    # init_for_distributed(args)
    # local_gpu_id = args.gpu

    ############################ data loder
    config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)


    torch.set_num_threads(os.cpu_count())

    if not os.path.exists('result/' + args.output): os.makedirs('result/' + args.output)
    dset = dataset_h5_v1()
    for sampleInfo in config['samples']:
        if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
        name = sampleInfo['name']
        dset.addSample(name, sampleInfo['path'])
    dset.initialize(args.output)

    lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
    lengths.append(len(dset)-sum(lengths))
    torch.manual_seed(config['training']['randomSeed'])
    trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)

    # trnLoader = DataLoader(trnDset, **kwargs)
    train_sampler = DistributedSampler(trnDset,shuffle=True)
    trnLoader = torch.utils.data.DataLoader(trnDset, batch_size=int(config['training']['batch']/args.world_size), shuffle=False, num_workers = config['training']['nDataLoaders'], pin_memory = True, sampler=train_sampler)

    # valLoader = DataLoader(valDset, **kwargs)
    val_sampler = DistributedSampler(valDset,shuffle=False)
    valLoader = torch.utils.data.DataLoader(valDset, batch_size=int(config['training']['batch']/args.world_size), shuffle=False, num_workers = config['training']['nDataLoaders'], pin_memory = True, sampler=val_sampler)
    torch.manual_seed(torch.initial_seed())

    ################# define model

    model = piver_mul(fea = config['model']['fea'], \
                    cla = config['model']['cla'], \
                    depths = config['model']['depths'], \
                    hidden = config['model']['hidden'], \
                    heads = config['model']['heads'], \
                    posfeed = config['model']['posfeed'], \
                    dropout = config['model']['dropout'], \
                    batch = int(config['training']['batch']/args.world_size), \
                    pmts = config['model']['pmts'], \
                    num_latents = config['model']['num_latents'], \
                    query_dim = config['model']['query_dim'], \
                    device= local_gpu_id)

    model.cuda(local_gpu_id)
    model = DistributedDataParallel(module = model, device_ids=[local_gpu_id], find_unused_parameters=True)


    torch.save(model, os.path.join('result/' + args.output, 'model.pth'))
    crit = LogCoshLoss().to(local_gpu_id)

    optm = torch.optim.Adam(model.parameters(), lr=config['training']['learningRate'])

    # scheduler = StepLR(optimizer=opim,step_size=30, gamma=0.1)



    bestState, bestLoss = {}, 1e9
    train = {'loss':[], 'val_loss':[]}
    nEpoch = config['training']['epoch']


    f = h5py.File("testh5.h5", 'r', libver='latest', swmr=True)
    pos = np.concatenate((np.array(f['geom']['pmt_x']).reshape(-1,1),np.array(f['geom']['pmt_y']).reshape(-1,1),np.array(f['geom']['pmt_z']).reshape(-1,1)),axis =1)
    pos = torch.Tensor(pos).unsqueeze(0).repeat(int(config['training']['batch']/args.world_size),1,1).to(local_gpu_id)
    f.close()

    for epoch in range(nEpoch):
        model.train()
        
        trn_loss, trn_acc = 0., 0.
        nProcessed = 0
        
        
        train_sampler.set_epoch(epoch)

        for i, (data,label,mask) in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):

            
            data = torch.Tensor(data).to(local_gpu_id)
            mask = torch.Tensor(mask).to(local_gpu_id)
            
            
            labels = torch.Tensor(label).float().to(device=local_gpu_id) ### vertex
            
            label = labels.reshape(-1,3)

            pred = model(data,pos,mask)

            
            loss = crit(pred, label)
            loss.backward()
            optm.step()
            optm.zero_grad()


            ibatch = len(label)
            nProcessed += ibatch
            trn_loss += loss.item()*ibatch

            

        trn_loss /= nProcessed 

        print(trn_loss,'trn_loss')

        model.eval()
        val_loss, val_acc = 0., 0.
        nProcessed = 0
        for i, (data,label,mask) in enumerate(tqdm(valLoader)):

            data = data.to(local_gpu_id)
            mask = mask.to(local_gpu_id)

            labels = label.float().to(device=local_gpu_id)

            


            label = labels.reshape(-1,3)

            pred = model(data,pos,mask)


            
            loss = crit(pred, label)
            
            ibatch = len(label)
            nProcessed += ibatch
            val_loss += loss.item()*ibatch

                
        val_loss /= nProcessed
        print(val_loss,'val_loss')
        if bestLoss > val_loss:
            bestState = model.to('cpu').state_dict()
            bestLoss = val_loss
            torch.save(bestState, os.path.join('result/' + args.output, 'weight.pth'))

            model.to(local_gpu_id)
        
        
        train['loss'].append(trn_loss)
        train['val_loss'].append(val_loss)

        with open(os.path.join('result/' + args.output, 'train.csv'), 'w') as f:
            writer = csv.writer(f)
            keys = train.keys()
            writer.writerow(keys)
            for row in zip(*[train[key] for key in keys]):
                writer.writerow(row)
        
        


    bestState = model.to('cpu').state_dict()
    torch.save(bestState, os.path.join('result/' + args.output, 'weightFinal.pth'))
    return 0




def init_for_distributed(rank,args):
    
    # 1. setting for distributed training
    args.rank = rank
    local_gpu_id = int(args.device[args.rank])
    torch.cuda.set_device(local_gpu_id)
    if args.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id),args.rank)

    # 2. init_process_group
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=args.world_size,
                            rank=args.rank)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    setup_for_distributed(args.rank == 0)
    print(args,'--------------initialize--------------')
    return local_gpu_id


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

if __name__ == '__main__':
    parser = argparse.ArgumentParser('training', parents=[get_args_parser()])
    args = parser.parse_args()

    args.world_size = len(args.device)
    # args.num_workers = len(args.device) * 20
    
    
    
    mp.spawn(main_worker, args=(args,),nprocs=args.world_size,join=True)


