import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


search_h5 = os.path.join('/store/cpnr/users/yewzzang/KNO_mu_500/h5_wall2',"*.h5")
# search_h5 = os.path.join('/store/cpnr/users/yewzzang/KNO_cla_e/h5',"*.h5")
files = glob.glob(search_h5)

for i in range(len(files)):
    f = h5py.File(files[i],'r')
    print(np.array(f['event']['pmt_q']).sum(), np.array(f['event']['pmt_t']).sum(),i, files[i])
    