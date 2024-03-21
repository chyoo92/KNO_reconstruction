import os
import ROOT
import numpy as np
import h5py
from tqdm import tqdm
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, required=True, help='Path to input directory')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
args = parser.parse_args()

file_name = args.input
output_file = args.output

f = h5py.File(file_name, 'r', libver='latest', swmr=True)
aaa = np.array((0,0,0))



pmt_q_empty = np.empty((0,30912))
pmt_t_empty = np.empty((0,30912))

vtx_x_empty = np.empty((0,1))
vtx_y_empty = np.empty((0,1))
vtx_z_empty = np.empty((0,1))

for i in tqdm(range(len(f['event']['pmt_q']))):
# for i in range(10):
    a = np.array(f['event']['pmt_q'])[i][np.array(f['event']['pmt_q'][i])>0]
    cen_dis = [np.array(f['event']['vtx_x'])[i],np.array(f['event']['vtx_y'])[i],np.array(f['event']['vtx_z'])[i]]

    if (len(a)>500)&(np.linalg.norm(aaa-np.array(cen_dis))<3500):
        pmt_q_empty = np.concatenate((pmt_q_empty,np.array(f['event']['pmt_q'])[i].reshape(-1,30912)),axis=0)
        pmt_t_empty = np.concatenate((pmt_t_empty,np.array(f['event']['pmt_t'])[i].reshape(-1,30912)),axis=0)

        vtx_x_empty = np.concatenate((vtx_x_empty,np.array(f['event']['vtx_x'])[i].reshape(-1,1)),axis=0)
        vtx_y_empty = np.concatenate((vtx_y_empty,np.array(f['event']['vtx_y'])[i].reshape(-1,1)),axis=0)
        vtx_z_empty = np.concatenate((vtx_z_empty,np.array(f['event']['vtx_z'])[i].reshape(-1,1)),axis=0)
    




kwargs = {'dtype':'f4', 'compression':'lzf'}
with h5py.File(output_file, 'w', libver='latest') as fout:
    gGeom = fout.create_group('geom')
    gGeom.create_dataset('pmt_x', data=np.array(f['geom']['pmt_x']), **kwargs)
    gGeom.create_dataset('pmt_y', data=np.array(f['geom']['pmt_y']), **kwargs)
    gGeom.create_dataset('pmt_z', data=np.array(f['geom']['pmt_z']), **kwargs)


    gEvent = fout.create_group('event')
    gEvent.create_dataset('vtx_x', data=vtx_x_empty, **kwargs)
    gEvent.create_dataset('vtx_y', data=vtx_y_empty, **kwargs)
    gEvent.create_dataset('vtx_z', data=vtx_z_empty, **kwargs)

    gEvent.create_dataset('pmt_q', data=pmt_q_empty, **kwargs)
    gEvent.create_dataset('pmt_t', data=pmt_t_empty, **kwargs)
        
