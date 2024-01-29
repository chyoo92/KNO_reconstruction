import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch_geometric.data import InMemoryDataset as PyGDataset, Data as PyGData
from bisect import bisect_right
from glob import glob
import numpy as np
import math
class dataset(PyGDataset):
    def __init__(self, **kwargs):
        super(dataset, self).__init__(None, transform=None, pre_transform=None)
        self.isLoaded = False

        self.fNames = []
        self.sampleInfo = pd.DataFrame(columns=["procName", "fileName", "fileIdx"])

    def len(self):
        return int(self.maxEventsList[-1])

    def get(self, idx):
        if not self.isLoaded: self.initialize()

        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]
        idx = int(idx - offset)



        data = self.dataList[fileIdx][idx]
         

        return data
    
    def addSample(self, procName, fNamePattern, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fNames))
        print(procName, fNamePattern)

        for fName in glob(fNamePattern):
#           
            fileIdx = len(self.fNames)
            self.fNames.append(fName)
            info = {'procName':procName, 'nEvent':0, 'fileName':fName, 'fileIdx':fileIdx}
            self.sampleInfo = self.sampleInfo.append(info, ignore_index=True)
         
    ### data load and remake
    def initialize(self, tev, output):
        if self.isLoaded: return

        
        procNames = list(self.sampleInfo['procName'].unique())  ## config file 'name'
        
        ### all file empty list
        self.procList, self.dataList = [], []
                
        ### file num check
        nFiles = len(self.sampleInfo)
        
        
        ### Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):
            
            ### file load and check event num
            f = torch.load(fName)

            nEvents = len(f)

            
            self.sampleInfo.loc[i, 'nEvent'] = nEvents

            self.dataList.append(f)

        

            procIdx = procNames.index(self.sampleInfo['procName'][i])
            self.procList.append(torch.ones(nEvents, dtype=torch.int32, requires_grad=False)*procIdx)


            
        ### save sampleInfo file in train result path
        SI = self.sampleInfo
        if tev == 1:
            SI.to_csv('result/'+output + '/training_sampleInfo.csv')
        else:
            SI.to_csv('result/'+output + '/evaluation_sampleInfo.csv')
        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvent']))) 

        self.isLoaded = True
