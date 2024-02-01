import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch_geometric.data import InMemoryDataset as PyGDataset, Data as PyGData
from bisect import bisect_right
from glob import glob
import numpy as np
import math
from tqdm import tqdm


def load_file(file_name,idx,maxpmts):
    f = torch.load(file_name)
    events = f[idx]

    if events.x.shape[0]<maxpmts:
        fea = torch.zeros(maxpmts,2)
        pos = torch.zeros(maxpmts,3)
        mask = torch.ones(maxpmts,5)

        fea[:events.x.shape[0],:] = events.x
        pos[:events.x.shape[0],:] = events.pos
        mask[:events.x.shape[0],:] = 0
        label = events.y
    else:
        fea = events.x
        pos = events.pos
        mask = torch.zeros(maxpmts,5)
        label = events.y

    return fea, pos, mask, label

class dataset_v5(PyGDataset):
    def __init__(self, **kwargs):
        super(dataset_v5, self).__init__(None, transform=None, pre_transform=None)
        self.isLoaded = False

        self.fNames = []
        self.sampleInfo = pd.DataFrame(columns=["procName", "fileName", "fileIdx"])



    def __getitem__(self, idx):
        if not self.isLoaded: self.initialize()

        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]
        idx = int(idx - offset)
        maxpmts = self.maxpmts
 

        fea, pos, mask, label = load_file(self.sampleInfo['fileName'][fileIdx],idx,maxpmts)



        return fea, label, mask, pos
    
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

    ### data load and remake
    def initialize(self, output):
        if self.isLoaded: return

        
        procNames = list(self.sampleInfo['procName'].unique())  ## config file 'name'
        
        ### all file empty list
        self.procList, self.dataList = [], []
        self.feaList, self.labelList, self.maskList, self.posList = [], [], [], []                
        ### file num check
        nFiles = len(self.sampleInfo)
        
        
        ### Load event contents
        
        ######## find max pmts################################
        max_pmts = 0
        print('-----find max pmts----')
        for i, fName in tqdm(enumerate(self.sampleInfo['fileName'])):
            
            ### file load and check event num
            f = torch.load(fName)
            nEvents = len(f)
            self.sampleInfo.loc[i, 'nEvent'] = nEvents
            
            for j in range(nEvents):
                # print(f[j].x.shape)
                if f[j].x.shape[0] > max_pmts:
                    max_pmts = f[j].x.shape[0]

            procIdx = procNames.index(self.sampleInfo['procName'][i])
            self.procList.append(torch.ones(nEvents, dtype=torch.int32, requires_grad=False)*procIdx)
        self.maxpmts = max_pmts
        print('-------max pmts = ' + str(max_pmts) + '------')
        ########################################################

        ### save sampleInfo file in train result path
        SI = self.sampleInfo
        SI.to_csv('result/'+output + '/sampleInfo.csv')


        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvent']))) 

        self.isLoaded = True

    def len(self):
        return int(self.maxEventsList[-1])