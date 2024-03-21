#!/usr/bin/env python
import numpy as np
import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import torch_geometric.nn as PyG
# from torch_geometric.transforms import Distance
# from torch_geometric.data import DataLoader
# from torch_geometric.data import Data as PyGData
# from torch_geometric.data import Data
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
from model.allModel import *
from datasets.dataset_pid_h5 import *
dset = dataset_pid_h5()

parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
args = parser.parse_args()

config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)

sys.path.append("./python")

torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)



for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(name, sampleInfo['path'])
    dset.setProcessLabel(name, sampleInfo['label'])

dset.initialize(args.output)




lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed'])
kwargs = {'num_workers':min(config['training']['nDataLoaders'], os.cpu_count()),
        'batch_size':config['training']['batch'], 'pin_memory':False}

trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)
testLoader = DataLoader(testDset, **kwargs)
torch.manual_seed(torch.initial_seed())



device = 'cpu'
model = torch.load('result/' + args.output+'/model.pth', map_location=device)
model.load_state_dict(torch.load('result/' + args.output+'/weight.pth', map_location=device))


dd = 'result/' + args.output + '/train.csv'

dff = pd.read_csv(dd)



##### Start evaluation #####
from tqdm import tqdm
label_s, preds = [], []
weights = []
scaledWeights = []
procIdxs = []
fileIdxs = []
idxs = []
features = []
batch_size = []
real_weights = []
scales = []
jade_label = []
eval_resampling = []
eval_real = []
model.eval()
ens = []
val_loss, val_acc = 0., 0.

for i, (fea, label, pos) in enumerate(tqdm(testLoader)):

    data = fea.to(device)
    pos = pos.to(device)

    label = label.float().to(device=device) ### vertex
    


    pred = model(data,pos)
    pred = torch.sigmoid(pred)
  
    label_s.extend([x.item() for x in label.view(-1)])
    preds.extend([x.item() for x in pred.view(-1)])
    


df = pd.DataFrame({'prediction':preds, 'label':label_s})
fPred = 'result/' + args.output + '/' + args.output + '.csv'
df.to_csv(fPred, index=False)