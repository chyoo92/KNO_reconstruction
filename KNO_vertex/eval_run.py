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
import matplotlib.pyplot as plt
import matplotlib.tri as tri

sys.path.append("./module")

parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output file')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
args = parser.parse_args()

config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)



torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)

##### Define dataset instance #####
from datasets.dataset_h5_v4 import NeuEvDataset as Dataset
dset = Dataset()


for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(sampleInfo['path'])
    
dset.initialize()


lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed'])
kwargs = {'num_workers':min(config['training']['nDataLoaders'],os.cpu_count()), 'pin_memory':False}


trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)
# testLoader = torch.utils.data.DataLoader(testDset, batch_size=100, shuffle=False, **kwargs)
testLoader = torch.utils.data.DataLoader(testDset, batch_size=config['training']['batch'], shuffle=False, **kwargs)
torch.manual_seed(torch.initial_seed())

##### Define model instance #####
from model.allModel import *


device = 'cuda:'+str(args.device)
model = torch.load('result/' + args.output+'/model.pth',map_location='cuda')
model.load_state_dict(torch.load('result/' + args.output+'/minval_model_module_state_dict.pth',map_location='cuda'),strict=False)
model = model.cuda(device)



dd = 'result/' + args.output + '/train.csv'

dff = pd.read_csv(dd)

pmt_pos_pre = dset.pmt_pos.to(device)

##### Start evaluation #####
from tqdm import tqdm
label_s, preds = [], []
weights = []

model.eval()

val_loss, val_acc = 0., 0.

for i, (pmt_q, pmt_t, vtx_pos,pmt_pos) in enumerate(tqdm(testLoader)):
    
    pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
    pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
    
    pmt_pos = pmt_pos.to(device)
            
            
    data = torch.cat([pmts_q,pmts_t],dim=2)
    labels = vtx_pos.float().to(device=device) ### vertex
    
    label = labels.reshape(-1,3)
    pred = model(data,pmt_pos)

        
    label_s.extend([x.item() for x in labels.view(-1)])
    preds.extend([x.item() for x in pred.view(-1)])

# for i, (pmt_q, pmt_t, vtx_pos) in enumerate(tqdm(testLoader)):
    
#     pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
#     pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(device)
    
#     data = torch.cat([pmts_q,pmts_t],dim=2)


#     pmt_pos = pmt_pos_pre.unsqueeze(0).repeat(data.shape[0],1,1).to(device)
    
#     labels = vtx_pos.float().to(device=device) ### vertex
    
#     label = labels.reshape(-1,3)
#     pred = model(data,pmt_pos)

        
#     label_s.extend([x.item() for x in labels.view(-1)])
#     preds.extend([x.item() for x in pred.view(-1)])   


df = pd.DataFrame({'prediction':preds, 'label':label_s})
fPred = 'result/' + args.output + '/' + args.output + '.csv'
df.to_csv(fPred, index=False)