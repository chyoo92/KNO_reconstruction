#!/usr/bin/env python
import numpy as np
import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

import torchdata.datapipes as dp
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService


sys.path.append("./module")
from model.allModel import *
from datasets.dataset_h5_v2 import NeuEvDataset as Dataset
dset = Dataset()

parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
args = parser.parse_args()

config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)


torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)
if not os.path.exists('result/' + args.output): os.makedirs('result/' + args.output)




##### Define dataset instance #####


for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(sampleInfo['path'])
dset.initialize()


lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed'])
trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)


kwargs = {'num_workers':min(config['training']['nDataLoaders'],os.cpu_count()), 'pin_memory':True}


trnLoader = DataLoader2(trnDset, batch_size=config['training']['batch'], shuffle=True, **kwargs,prefetch_factor=2, persistent_workers=True)
valLoader = DataLoader2(valDset, batch_size=config['training']['batch'], shuffle=False, **kwargs,prefetch_factor=2, persistent_workers=True)


# trnLoader = DataLoader(trnDset, batch_size=config['training']['batch'], shuffle=True, **kwargs,prefetch_factor=2, persistent_workers=True)
# valLoader = DataLoader(valDset, batch_size=config['training']['batch'], shuffle=False, **kwargs,prefetch_factor=2, persistent_workers=True)
torch.manual_seed(torch.initial_seed())



model_y = config['model']['model']
fea_y = config['model']['fea']
cla_y = config['model']['cla']
depths_y = config['model']['depths']
hidden_y = config['model']['hidden']
heads_y = config['model']['heads']
posfeed_y = config['model']['posfeed']
dropout_y = config['model']['dropout']
batch_y = config['training']['batch']
pmts_y = config['model']['pmts']
num_latents_y = config['model']['num_latents']
query_dim_y = config['model']['query_dim']

# #### Define model instance #####
exec('model = '+config['model']['model']+'(fea = fea_y, \
                            cla = cla_y, \
                            depths = depths_y, \
                            hidden = hidden_y, \
                            heads = heads_y, \
                            posfeed = posfeed_y, \
                            dropout = dropout_y, \
                            batch = batch_y, \
                            pmts = pmts_y, \
                            num_latents = num_latents_y, \
                            query_dim = query_dim_y, \
                            device= args.device)')
 

torch.save(model, os.path.join('result/' + args.output, 'model.pth'))


device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'


optm = torch.optim.Adam(model.parameters(), lr=config['training']['learningRate'])

pmt_pos_pre = dset.pmt_pos.to(device)
    
from sklearn.metrics import accuracy_score
from tqdm import tqdm
bestState, bestLoss = {}, 1e9
train = {'loss':[], 'val_loss':[]}
nEpoch = config['training']['epoch']



for epoch in range(nEpoch):
    model.train()
    
    trn_loss, trn_acc = 0., 0.
    nProcessed = 0
    
    

    for i, (pmt_q, pmt_t, vtx_pos) in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):
    # for i, (pmt_q, pmt_t, vtx_pos) in enumerate(trnLoader):


        pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
        pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)

        
        data = torch.cat([pmts_q,pmts_t],dim=2)



        pmt_pos = pmt_pos_pre.unsqueeze(0).repeat(data.shape[0],1,1).to(device)

        labels = vtx_pos.float().to(device=device) ### vertex
        
        label = labels.reshape(-1,3)

        pred = model(data,pmt_pos)
        crit = LogCoshLoss()



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
    for i, (pmt_q, pmt_t, vtx_pos) in enumerate(tqdm(valLoader)):
    # for i, (pmt_q, pmt_t, vtx_pos) in enumerate(valLoader):

        pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
        pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
        
        data = torch.cat([pmts_q,pmts_t],dim=2)


        pmt_pos = pmt_pos_pre.unsqueeze(0).repeat(data.shape[0],1,1).to(device)
        
        labels = vtx_pos.float().to(device=device) ### vertex
        
        label = labels.reshape(-1,3)
        pred = model(data,pmt_pos)

        crit = LogCoshLoss()
        
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

        model.to(device)
    
    
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
