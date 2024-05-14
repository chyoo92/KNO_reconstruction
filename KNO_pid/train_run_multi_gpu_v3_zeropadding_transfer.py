#!/usr/bin/env python
import numpy as np
import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F

import sys, os
import subprocess
import csv, yaml
import math
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.optim as optim
import argparse
import pandas as pd

import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score
from tqdm import tqdm


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

sys.path.append("./module")
from model.allModel import *
from datasets.dataset_pid_h5_v4 import NeuEvDataset as Dataset


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', action='store', type=str)
    parser.add_argument('-o', '--output', action='store', type=str)
    parser.add_argument('--device', nargs="+", default=['0','1','2','3'])
    parser.add_argument('--loadmap', action='store', type=str)
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
    if rank==0:
        if not os.path.exists('result/' + args.output): os.makedirs('result/' + args.output)
    dset = Dataset()
    for sampleInfo in config['samples']:
        if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
        name = sampleInfo['name']
        dset.addSample(sampleInfo['path'],sampleInfo['label'])

    dset.initialize()

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

    model = torch.load('result/' + args.loadmap+'/model.pth',map_location='cuda')
    model.load_state_dict(torch.load('result/' + args.loadmap+'/model_module_state_dict_rt.pth',map_location='cuda'),strict=False)
    model = model.cuda(local_gpu_id)

    if rank==0:
        torch.save(model, os.path.join('result/' + args.output, 'model.pth'))
        
    model = DistributedDataParallel(module = model, device_ids=[local_gpu_id], find_unused_parameters=True)


    # torch.save(model, os.path.join('result/' + args.output, 'model.pth'))
    crit = torch.nn.BCEWithLogitsLoss().to(local_gpu_id)

    optm = torch.optim.Adam(model.parameters(), lr=config['training']['learningRate'])

    # scheduler = StepLR(optimizer=opim,step_size=30, gamma=0.1)

    pmt_pos_pre = dset.pmt_pos.to(local_gpu_id)

    bestState, bestLoss = {}, 1e9
    train = {'loss':[], 'val_loss':[],'acc':[],'val_acc':[]}
    nEpoch = config['training']['epoch']
    for epoch in range(nEpoch):
        model.train()
        
        trn_loss, trn_acc = 0., 0.
        nProcessed = 0
        

        train_sampler.set_epoch(epoch)

        for i, (pmt_q,pmt_t, label, pmt_pos) in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):

            pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(local_gpu_id)
            pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(local_gpu_id)

            pmt_pos = pmt_pos.to(local_gpu_id)
            
            data = torch.cat([pmts_q,pmts_t],dim=2)

            # pmt_pos = pmt_pos_pre.unsqueeze(0).repeat(data.shape[0],1,1).to(local_gpu_id)
            
            label = label.float().to(device=local_gpu_id) ### vertex
            


            pred = model(data,pmt_pos)
            pred = pred.reshape(-1)

            
            loss = crit(pred, label)
            loss.backward()
            optm.step()
            optm.zero_grad()


            ibatch = len(label)
            nProcessed += ibatch
            trn_loss += loss.item()*ibatch
            trn_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0))*ibatch
            del data, label, pred, pmt_pos

        trn_loss /= nProcessed 
        trn_acc  /= nProcessed

        print(trn_loss,'trn_loss')
        print(trn_acc,'trn_acc')
        torch.save(model.state_dict(), os.path.join('result/' + args.output, 'model_state_dict_rt.pth'))
        torch.save(model.module.state_dict(), os.path.join('result/' + args.output, 'model_module_state_dict_rt.pth'))    
        
        model.eval()
        val_loss, val_acc = 0., 0.
        nProcessed = 0
        with torch.no_grad():
            for i, (pmt_q,pmt_t, label, pmt_pos) in enumerate(tqdm(valLoader)):

                pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(local_gpu_id)
                pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(local_gpu_id)

                
                pmt_pos = pmt_pos.to(local_gpu_id)
            
                
                data = torch.cat([pmts_q,pmts_t],dim=2)

                # pmt_pos = pmt_pos_pre.unsqueeze(0).repeat(data.shape[0],1,1).to(local_gpu_id)
                
                label = label.float().to(device=local_gpu_id) ### vertex
                

                

                pred = model(data,pmt_pos)
                pred = pred.reshape(-1)


                
                loss = crit(pred, label)
                
                ibatch = len(label)
                nProcessed += ibatch
                val_loss += loss.item()*ibatch
                val_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0))*ibatch
                del data, label, pred, pmt_pos
                    
                    
            val_loss /= nProcessed
            val_acc /= nProcessed
            print(val_loss,'val_loss')
            print(val_acc,'val_acc')
            if bestLoss > val_loss:
                bestState = model.to('cpu').state_dict()
                bestLoss = val_loss
                torch.save(bestState, os.path.join('result/' + args.output, 'weight.pth'))

                model.to(local_gpu_id)
                torch.save(model.module.state_dict(), os.path.join('result/' + args.output, 'model_scrpited_min_val.pth'))   
            
            
            train['loss'].append(trn_loss)
            train['acc'].append(trn_acc)
            train['val_loss'].append(val_loss)
            train['val_acc'].append(val_acc)

            with open(os.path.join('result/' + args.output, 'train.csv'), 'w') as f:
                writer = csv.writer(f)
                keys = train.keys()
                writer.writerow(keys)
                for row in zip(*[train[key] for key in keys]):
                    writer.writerow(row)
    
        


    bestState = model.to('cpu').state_dict()
    torch.save(bestState, os.path.join('result/' + args.output, 'weightFinal.pth'))
    torch.save(model.module.state_dict(), os.path.join('result/' + args.output, 'model_final.pth'))    
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
    args.num_workers = len(args.device) * 0
    
    
    
    mp.spawn(main_worker, args=(args,),nprocs=args.world_size,join=True)


