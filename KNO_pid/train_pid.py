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


sys.path.append("./module")
from model.allModel import *
from datasets.dataset_pid import *
dset = dataset_pid()


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
    dset.addSample(name, sampleInfo['path'])
    dset.setProcessLabel(name, sampleInfo['label'])
dset.initialize(args.output)


lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed'])
trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)


kwargs = {'num_workers':min(config['training']['nDataLoaders'],os.cpu_count()), 'pin_memory':True}

trnLoader = DataLoader(trnDset, batch_size=config['training']['batch'], shuffle=True, **kwargs)
valLoader = DataLoader(valDset, batch_size=config['training']['batch'], shuffle=False, **kwargs)
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


    
from sklearn.metrics import accuracy_score
from tqdm import tqdm
bestState, bestLoss = {}, 1e9
train = {'loss':[], 'val_loss':[],'acc':[],'val_acc':[]}
nEpoch = config['training']['epoch']
for epoch in range(nEpoch):
    model.train()
    
    trn_loss, trn_acc = 0., 0.
    nProcessed = 0
    

    
    for i, (data,label,mask,pos) in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):

        
        data = data.to(device)
        mask = mask.to(device)
        pos = pos.to(device)
        
        label = label.float().to(device=device) ### vertex
        


        pred = model(data,pos,mask)
        pred = pred.reshape(-1)

        crit = torch.nn.BCEWithLogitsLoss()
        
        loss = crit(pred, label)
        loss.backward()
        optm.step()
        optm.zero_grad()




        ibatch = len(label)
        nProcessed += ibatch
        trn_loss += loss.item()*ibatch
        trn_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0))*ibatch

        

    trn_loss /= nProcessed 
    trn_acc  /= nProcessed

    print(trn_loss,'trn_loss')
    print(trn_acc,'trn_acc')



    model.eval()
    val_loss, val_acc = 0., 0.
    nProcessed = 0
    for i, (data,label,mask,pos) in enumerate(tqdm(valLoader)):

        data = data.to(device)
        mask = mask.to(device)
        pos = pos.to(device)
        label = label.float().to(device=device)
        

        




        pred = model(data,pos,mask)
        pred = pred.reshape(-1)


        crit = torch.nn.BCEWithLogitsLoss()
        
        loss = crit(pred, label)
        
        ibatch = len(label)
        nProcessed += ibatch
        val_loss += loss.item()*ibatch
        val_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0))*ibatch

            
            
    val_loss /= nProcessed
    val_acc /= nProcessed
    print(val_loss,'val_loss')
    print(val_acc,'val_acc')
    if bestLoss > val_loss:
        bestState = model.to('cpu').state_dict()
        bestLoss = val_loss
        torch.save(bestState, os.path.join('result/' + args.output, 'weight.pth'))

        model.to(device)
    
    
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
