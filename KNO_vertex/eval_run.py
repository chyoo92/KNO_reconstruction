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
from datasets.dataset_v2 import *
dset = dataset_v2()

for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(name, sampleInfo['path'])
    
dset.initialize(args.output)



lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed'])
kwargs = {'num_workers':min(config['training']['nDataLoaders'],os.cpu_count()), 'pin_memory':False}

trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)
testLoader = DataLoader(testDset, batch_size=config['training']['batch'], shuffle=False, **kwargs)
torch.manual_seed(torch.initial_seed())

##### Define model instance #####
from model.allModel import *



device = 'cpu'
model = torch.load('result/' + args.output+'/model.pth', map_location='cpu')
model.load_state_dict(torch.load('result/' + args.output+'/weight.pth', map_location='cpu'))

if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'
    model = torch.load('result/' + args.output+'/model.pth', map_location=device)
    model.load_state_dict(torch.load('result/' + args.output+'/weight.pth', map_location=device))

dd = 'result/' + args.output + '/train.csv'

dff = pd.read_csv(dd)



##### Start evaluation #####
from tqdm import tqdm
label_s, preds = [], []
weights = []

model.eval()

val_loss, val_acc = 0., 0.

for i, (data,label,mask,pos) in enumerate(tqdm(testLoader)):
    
    data = data.to(device)
    mask = mask.to(device)
    pos = pos.to(device)
    
    labels = label.float().to(device=device) ### vertex
    
    label = labels.reshape(-1,3)

    
    pred = model(data,pos,mask)
    
        
    label_s.extend([x.item() for x in labels.view(-1)])
    preds.extend([x.item() for x in pred.view(-1)])
    


df = pd.DataFrame({'prediction':preds, 'label':label_s})
fPred = 'result/' + args.output + '/' + args.output + '.csv'
df.to_csv(fPred, index=False)