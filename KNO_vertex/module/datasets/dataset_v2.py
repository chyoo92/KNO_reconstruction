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

class dataset_v2(PyGDataset):
    def __init__(self, **kwargs):
        super(dataset_v2, self).__init__(None, transform=None, pre_transform=None)
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
        
        fea_List = self.feaList[fileIdx][idx]
        label_List = self.labelList[fileIdx][idx]
        mask_List = self.maskList[fileIdx][idx]
        pos_List = self.posList[fileIdx][idx]



        # data = self.dataList[fileIdx][idx]
         

        return fea_List, label_List, mask_List, pos_List
    
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
            
            for j in range(nEvents):
                # print(f[j].x.shape)
                if f[j].x.shape[0] > max_pmts:
                    max_pmts = f[j].x.shape[0]
        print('-------max pmts = ' + str(max_pmts) + '------')
        ########################################################

        for i, fName in tqdm(enumerate(self.sampleInfo['fileName'])):
            
            ### file load and check event num
            f = torch.load(fName)
            print(fName, i/len(self.sampleInfo['fileName']))
            nEvents = len(f)
            
            feas = []
            poss = []
            labels = []
            masks = []

            
            self.sampleInfo.loc[i, 'nEvent'] = nEvents
            for events in range(nEvents):
                if f[events].x.shape[0]<max_pmts:
                    fea = torch.zeros(max_pmts,2)
                    pos = torch.zeros(max_pmts,3)
                    mask = torch.ones(max_pmts,5)

                    fea[:f[events].x.shape[0],:] = f[events].x
                    pos[:f[events].x.shape[0],:] = f[events].pos
                    
                    mask[:f[events].x.shape[0],:] = 0


                    feas.append(fea)
                    poss.append(pos)
                    labels.append(f[events].y)
                    masks.append(mask)

                    
                    # print(fea.shape)
                    # print(label.shape)
                    # print(mask.shape)
                else:

                    feas.append(f[events].x)
                    poss.append(f[events].pos)
                    masks.append(torch.zeros(max_pmts,5))
                    labels.append(f[events].y)




                    # print(fea.shape)
                    # print(label.shape)
                    # print(mask.shape)

            
            self.feaList.append(feas)
            self.labelList.append(labels)
            self.maskList.append(masks)
            self.posList.append(poss)
            feas,poss,labels,masks=[],[],[],[]
            # self.dataList.append(f)
        

        

            procIdx = procNames.index(self.sampleInfo['procName'][i])
            self.procList.append(torch.ones(nEvents, dtype=torch.int32, requires_grad=False)*procIdx)


        print('---------total events = ' + str(len(feas)) +'------')
        ### save sampleInfo file in train result path
        SI = self.sampleInfo
        SI.to_csv('result/'+output + '/sampleInfo.csv')


        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvent']))) 

        self.isLoaded = True
