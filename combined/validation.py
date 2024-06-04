#!/usr/bin/env python
import h5py
import sys, os
import numpy as np
import csv, yaml
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import ast
import subprocess
import math
from loss_functions import *

## torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## torch multi-gpu
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

sys.path.append("./module")

from model.allModel import *
from datasets import dataset_main


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', action='store', type=str)
    parser.add_argument('-o', '--output', action='store', type=str)
    parser.add_argument('--type', action='store', type=int, default=0) ####### 0 vertex / 1 pid
    parser.add_argument('--padding', action='store', type=int, default=0) ####### 0 no padding / 1 charge & time 0 to vtx zeropadding
    parser.add_argument('--datasetversion', action='store', type=int, default=0) ####### 0 main / 1 test
    parser.add_argument('--device', action='store', type=int, default=None)
    parser.add_argument('--multi_device', type=int, default=2)
    parser.add_argument('--transfer_learning', action='store', type=int, default=0) #### 0 first training / 1 transfer learning
    parser.add_argument('--rank_i', type=int, default=0)

    parser.add_argument('--cla', action='store', type=int, default=3)

    #### training parameter
    parser.add_argument('--nDataLoaders', action='store', type=int, default=4)
    parser.add_argument('--epoch', action='store', type=int, default=300)
    parser.add_argument('--batch', action='store', type=int, default=25)
    parser.add_argument('--learningRate', action='store', type=float, default=0.001)
    parser.add_argument('--randomseed', action='store', type=int, default=12345)



    return parser


def main_one_gpu(args):
    ### select dataset version
    if args.datasetversion == 0:
        dataset_module = dataset_main  #### main dataset code
    elif args.datasetversion == 1:
        dataset_module = dataset_test  #### test dataset code
    Dataset = dataset_module.NeuEvDataset

    device = 'cuda'

    #### config file load
    config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
    if args.nDataLoaders: config['training']['nDataLoaders'] = args.nDataLoaders
    if args.epoch: config['training']['epoch'] = args.epoch
    if args.batch: config['training']['batch'] = args.batch
    if args.learningRate: config['training']['learningRate'] = args.learningRate
    if args.randomseed: config['training']['randomSeed'] = args.randomseed
    



    if args.type == 0:
        result_path = 'result_vtx/' + args.output
        
    elif args.type == 1:
        result_path = 'result_pid/' + args.output
        
    #### dataset 
    dset = Dataset()



    trnLoader, valLoader, testLoader = data_setting(args, config, dset)

    
    #### model load
    
    model = torch.load(result_path+'/model.pth',map_location='cuda')
    model.load_state_dict(torch.load(result_path+'/weight.pth',map_location='cuda'),strict=False)

    model = model.cuda(device)
    
    ###############################################
    ################## test #######################
    ###############################################


    labels, preds, fnames = [], [], []
    model.eval()
    if args.type == 0:
        for i, (pmt_q,pmt_t, label, pmt_pos, fName) in enumerate(tqdm(testLoader)):

            pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
            pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
            pmt_pos = pmt_pos.to(device)
            
            data = torch.cat([pmts_q,pmts_t],dim=2)

            label = label.float().to(device=device)
            if args.type == 0: label = label.reshape(-1,3)

            pred = model(data,pmt_pos)
            if args.type == 1: 
                pred = pred.reshape(-1)
                pred = torch.sigmoid(pred)

            labels.extend([x.item() for x in label.view(-1)])
            preds.extend([x.item() for x in pred.view(-1)])
            if args.type == 1:
                fnames.extend([x.item() for x in np.array(fName)])

            del pmts_q, pmt_t, pmt_pos, data, label, pred, fName
    elif args.type == 1:
        for i, (pmt_q,pmt_t, label, pmt_pos, fName) in enumerate(tqdm(testLoader)):

            pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
            pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
            pmt_pos = pmt_pos.to(device)
            
            data = torch.cat([pmts_q,pmts_t],dim=2)

            label = label.float().to(device=device)
            if args.type == 0: label = label.reshape(-1,3)

            pred = model(data,pmt_pos)
            if args.type == 1: 
                pred = pred.reshape(-1)
                pred = torch.sigmoid(pred)

            labels.extend([x.item() for x in label.view(-1)])
            preds.extend([x.item() for x in pred.view(-1)])
            if args.type == 1:
                fnames.extend([x.item() for x in np.array(fName)])

            del pmts_q, pmt_t, pmt_pos, data, label, pred, fName

    if args.type == 0:
        df = pd.DataFrame({'prediction':preds, 'label':labels})
    elif args.type == 1:
        df = pd.DataFrame({'prediction':preds, 'label':labels,'fname':fnames})

    fPred = result_path+'/' + args.output + '.csv'
    df.to_csv(fPred, index=False)

    del preds, labels, fnames



    return 0






##########################################################
################## Setting data ##########################
##########################################################


def data_setting(args,config,dset):


    for sampleInfo in config['samples']:
        if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
        name = sampleInfo['name']
        dset.addSample(sampleInfo['path'], sampleInfo['label'],args.type, args.padding)
    dset.initialize() 


    #### split events
    lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
    lengths.append(len(dset)-sum(lengths))
    torch.manual_seed(config['training']['randomSeed'])
    trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)

    
    kwargs = {'batch_size':config['training']['batch'],'num_workers':min(config['training']['nDataLoaders'],os.cpu_count()), 'pin_memory':True}

    trnLoader = torch.utils.data.DataLoader(trnDset, shuffle=True, **kwargs)
    valLoader = torch.utils.data.DataLoader(valDset, shuffle=False, **kwargs)
    testLoader = torch.utils.data.DataLoader(testDset, shuffle=False, **kwargs)
    torch.manual_seed(torch.initial_seed())

    return trnLoader, valLoader, testLoader    


##########################################################
###################### main running ######################
##########################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser('training', parents=[get_args_parser()])
    args = parser.parse_args()


    
    args.device = args.device

    with open(args.output, "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    main_one_gpu(args)

        
