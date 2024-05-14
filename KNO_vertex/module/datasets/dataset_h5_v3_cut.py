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
        self.cEvents = None ## Cumulative sum of nEvents in each file

        self.pmt_pos = None
        # self.pmt_dir = None
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

        pmt_q = torch.FloatTensor(fin['event/pmt_q'][ii])
        pmt_t = torch.FloatTensor(fin['event/pmt_t'][ii])

        # pmt_q = np.array(fin['event/pmt_q'][ii])
        # pmt_t = np.array(fin['event/pmt_t'][ii])

        vtx_pos = torch.zeros(3)
        # vtx_dir = torch.zeros(3)
        #vtx_ke = torch.zeros(1)
        if self.isMC:
            vtx_x = fin['event/vtx_x'][ii]
            vtx_y = fin['event/vtx_y'][ii]
            vtx_z = fin['event/vtx_z'][ii]
            # vtx_pos = torch.FloatTensor([vtx_x, vtx_y, vtx_z])
            vtx_pos = np.array([vtx_x, vtx_y, vtx_z])
            
            
            #vtx_t = fin['event/vtx_t'][ii]

            # vtx_px = fin['event/vtx_px'][ii]
            # vtx_py = fin['event/vtx_py'][ii]
            # vtx_pz = fin['event/vtx_pz'][ii]
            # vtx_dir = torch.FloatTensor([vtx_px, vtx_py, vtx_pz])
            #vtx_ke = fin['event/vtx_ke'][ii]

        #return pmt_q, pmt_t, pmt_pos, vtx_pos
        # return pmt_q, pmt_t, vtx_pos/self.posScale
        return pmt_q, pmt_t, torch.Tensor(vtx_pos)
        #return pmt_q, pmt_t, vtx_pos/self.posScale, vtx_dir, vtx_ke

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

            # pmt_x = np.array(geom['pmt_x'])
            # pmt_y = np.array(geom['pmt_y'])
            # pmt_z = np.array(geom['pmt_z'])
            # self.pmt_pos = np.stack([pmt_x, pmt_y, pmt_z], dim=1)

            # pmt_px = torch.FloatTensor(geom['pmt_px'])
            # pmt_py = torch.FloatTensor(geom['pmt_py'])
            # pmt_pz = torch.FloatTensor(geom['pmt_pz'])
            # self.pmt_dir = torch.stack([pmt_px, pmt_py, pmt_pz], dim=1)

            ## Find scaler
            # self.posScale = torch.abs(self.pmt_pos).max()
            # self.pmt_pos /= self.posScale

        self.isInit = True

    def addSample(self, fName):
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

        self.cEvents = np.cumsum(self.nEvents)
        self.nEventsTotal = self.cEvents[-1]