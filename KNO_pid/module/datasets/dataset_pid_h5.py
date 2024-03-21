import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# from torch_geometric.data import InMemoryDataset as PyGDataset, Data as PyGData
from bisect import bisect_right
from glob import glob
import numpy as np
import math
from tqdm import tqdm




class dataset_pid_h5(Dataset):
    def __init__(self, **kwargs):
        super(dataset_pid_h5, self).__init__()
        self.isLoaded = False

        self.fNames = []
        self.sampleInfo = pd.DataFrame(columns=["procName", "fileName", "fileIdx"])



    def __getitem__(self, idx):
        if not self.isLoaded: self.initialize()

        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]
        idx = int(idx - offset)

        pos = self.pmts_pos

        fea = self.feaList[fileIdx][idx]

        label = self.labelsList[fileIdx][idx]


        return torch.Tensor(fea), torch.Tensor(label), torch.Tensor(pos)

    
    def addSample(self, procName, fNamePattern, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fNames))
        print(procName, fNamePattern)

        for fName in glob(fNamePattern):
#           
            fileIdx = len(self.fNames)
            self.fNames.append(fName)
            info = {'procName':procName, 'nEvent':0, 'fileName':fName, 'fileIdx':fileIdx}
            # self.sampleInfo = self.sampleInfo.append(info, ignore_index=True)
            self.sampleInfo = pd.concat([self.sampleInfo,pd.DataFrame(info,index=[0])], ignore_index=True)
    def setProcessLabel(self, procName, label):
        self.sampleInfo.loc[self.sampleInfo.procName==procName, 'label'] = label
    ### data load and remake
    def initialize(self, output):
        if self.isLoaded: return

        
        procNames = list(self.sampleInfo['procName'].unique())  ## config file 'name'
        
        ### all file empty list
        self.procList, self.dataList = [], []
        self.pmts_pos = []
        self.feaList, self.labelsList, self.posList = [], [], []            
        ### file num check
        nFiles = len(self.sampleInfo)
        
        
        ### Load event contents

        for i, fName in tqdm(enumerate(self.sampleInfo['fileName'])):
            

            f = h5py.File(fName, 'r', libver='latest', swmr=True)

            nEvents = len(np.array(f['event']['pmt_q']))
            self.sampleInfo.loc[i, 'nEvent'] = nEvents
            label = self.sampleInfo['label'][i]
            labels = torch.ones(nEvents, dtype=torch.int32, requires_grad=False)*label
            self.labelsList.append(labels)
            
            fea = np.concatenate((np.array(f['event']['pmt_q']).reshape(-1,30912,1),np.array(f['event']['pmt_t']).reshape(-1,30912,1)),axis=2)
            self.feaList.append(fea)
            procIdx = procNames.index(self.sampleInfo['procName'][i])
            self.procList.append(torch.ones(nEvents, dtype=torch.int32, requires_grad=False)*procIdx)

            self.pmts_pos = np.concatenate((np.array(f['geom']['pmt_x']).reshape(-1,1),np.array(f['geom']['pmt_y']).reshape(-1,1),np.array(f['geom']['pmt_z']).reshape(-1,1)),axis =1)

            f.close()
        ### save sampleInfo file in train result path
        SI = self.sampleInfo
        SI.to_csv('result/'+output + '/sampleInfo.csv')


        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvent']))) 
    
        self.isLoaded = True

    def __len__(self):
        return int(self.maxEventsList[-1])