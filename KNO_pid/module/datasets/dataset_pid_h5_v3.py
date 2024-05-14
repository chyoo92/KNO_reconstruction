#!/usr/bin/env python
import torch
import numpy as np
import h5py
import os
from glob import glob

class NeuEvDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super().__init__()

        self.isInit = False

        ## self.isMC: MC truth information is available or not.
        ## will be adjusted automatically while reading the data
        self.isMC = True
        self.shape = None
        self.nEventsTotal = 0

        self.fNames = []
        self.nEvents = [] ## number of events in each file
        self.flabels = []
        self.cEvents = None ## Cumulative sum of nEvents in each file

        self.pmt_pos = None

        self.posScale = 1

    def __len__(self):
        return self.nEventsTotal

    def __str__(self):
        s = [
            " Summary of dataset",
            f"* nFiles  = {len(self.fNames)}",
            f"* nEvents = {self.nEventsTotal}",
            f"* isMC    = {self.isMC}",
            f"* isInit  = {self.isInit}",
        ]
        w = max([len(x) for x in s])+1
        s = ["-"*w, *s, "-"*w]
        return '\n'.join(s)

    def __getitem__(self, idx):
        if not self.isInit: self.initialize()
        fileIdx = torch.searchsorted(self.cEvents, idx)
        ii = idx-self.cEvents[fileIdx]

        fName = self.fNames[fileIdx]
        fin = h5py.File(fName, 'r', libver='latest', swmr=True)
        flabel = self.flabels[fileIdx]

        pmt_q = torch.FloatTensor(fin['event/pmt_q'][ii])
        pmt_t = torch.FloatTensor(fin['event/pmt_t'][ii])
        
        padding_index = (pmt_t>930)&(pmt_t<=1200)
        # print('---------------------------------------------------------------------------')        
        # print(pmt_q.shape)
        # print(pmt_t.shape)
        # print(padding_index.shape)
        # print(self.pmt_pos.shape)
        # print('---------------------------------------------------------------------------')
        pmt_q = pmt_q.masked_fill(~padding_index, 0)
        pmt_t = pmt_t.masked_fill(~padding_index, 0)
        # pmt_q = torch.where(padding_index.unsqueeze(dim=0), pmt_q, torch.zeros_like(pmt_q))
        # pmt_t = torch.where(padding_index.unsqueeze(dim=0), pmt_t, torch.zeros_like(pmt_t))
        pmt_pos = torch.where(padding_index.unsqueeze(dim=1), self.pmt_pos, torch.zeros_like(self.pmt_pos))
        
        # pmt_pos = torch.where(padding_index.unsqueeze(dim=1), self.pmt_pos, torch.zeros_like(self.pmt_pos))
        
        

        return pmt_q, pmt_t, flabel, pmt_pos

    def initialize(self):
        assert(self.isInit == False)

        self.nEvents = torch.tensor(self.nEvents, dtype=torch.int32)
        self.cEvents = torch.tensor(self.cEvents)

        with h5py.File(self.fNames[0]) as f:
            geom = f['geom']

            pmt_x = torch.FloatTensor(geom['pmt_x'])
            pmt_y = torch.FloatTensor(geom['pmt_y'])
            pmt_z = torch.FloatTensor(geom['pmt_z'])
            self.pmt_pos = torch.stack([pmt_x, pmt_y, pmt_z], dim=1)


        self.isInit = True


    def addSample(self, fName,flabel):
        ## Add samples for the given file name pattern

        ## For the case if directory is given to the argument
        if os.path.isdir(fName):
            fName = os.path.join(fName, "*.h5")

        ## Find hdf files and add to the list
        for fName in glob(fName):
            if not fName.endswith(".h5"): continue
            with h5py.File(fName) as f:
                if 'event' not in f: continue
                event = f['event']

                if 'pmt_q' not in event: continue
                nEvent = event["pmt_q"].shape[0] ## shape: (nEvents, pmt_N)
                self.shape = event["pmt_q"].shape[1:]

                if self.isMC and 'vtx_x' not in event:
                    ## assume the dataset is "real data"
                    ## if there's no vertex position information
                    self.isMC = False

                self.fNames.append(fName)
                self.nEvents.append(nEvent)
                self.flabels.append(flabel)

        self.cEvents = np.cumsum(self.nEvents)
        self.nEventsTotal = self.cEvents[-1]