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
from datasets.dataset_pid_h5_v5 import NeuEvDataset as Dataset
dset = Dataset()

parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
args = parser.parse_args()

config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)

sys.path.append("./python")

torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)



dset = Dataset()
for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(sampleInfo['path'],sampleInfo['label'])

dset.initialize()




lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed'])
kwargs = {'num_workers':min(config['training']['nDataLoaders'], os.cpu_count()),
        'batch_size':config['training']['batch'], 'pin_memory':False}

trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)
testLoader = DataLoader(testDset, **kwargs)
torch.manual_seed(torch.initial_seed())



model = torch.load('result/' + args.output+'/model.pth',map_location='cuda')
# model.load_state_dict(torch.load('result/' + args.output+'/weight.pth',map_location='cuda'))
model.load_state_dict(torch.load('result/' + args.output+'/model_scrpited_min_val.pth',map_location='cuda'),strict=False)
model = model.cuda(args.device)

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
fnames = []
val_loss, val_acc = 0., 0.

for i, (pmt_q,pmt_t, label, pmt_pos, fName) in enumerate(tqdm(testLoader)):
    
    pmts_q = pmt_q.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(args.device)
    pmts_t = pmt_t.reshape(pmt_q.shape[0],pmt_q.shape[1],1).to(args.device)

    pmt_pos = pmt_pos.to(args.device)

    data = torch.cat([pmts_q,pmts_t],dim=2)

            
    label = label.float().to(device=args.device) ### vertex
    


    pred = model(data,pmt_pos)
    pred = torch.sigmoid(pred)


    
    label_s.extend([x.item() for x in label.view(-1)])
    preds.extend([x.item() for x in pred.view(-1)])
    fnames.extend([x.item() for x in np.array(fName)])


df = pd.DataFrame({'prediction':preds, 'label':label_s,'fname':fnames})
fPred = 'result/' + args.output + '/' + args.output + '.csv'
df.to_csv(fPred, index=False)


df2 = pd.DataFrame({'fname':fnames})
f_name = 'result/' + args.output + '/' + args.output + '_fname.csv'
df2.to_csv(f_name, index=False)

