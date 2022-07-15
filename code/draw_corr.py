import mne
import numpy as np
import matplotlib
from collections import defaultdict
import mne
import scipy.signal
from mne.forward import make_forward_dipole
from mne.evoked import combine_evoked
from mne.simulation import simulate_evoked

import matplotlib.pyplot as plt
'''
file = 'cut_data/mfu1_11_data.npy'
data = np.load(file)
signal = data.T
channel = ['AF8', 'Fp2', 'Fp1', 'AF7', 'O1']
info = mne.create_info(channel , 250 , ch_types = "eeg")
# raw就是得到的结果了
raw = mne.io.RawArray(signal, info)
# raw.plot_psd(fmax=50)
raw_highpass = raw.copy().filter(l_freq=1, h_freq=None)

ica = ICA(n_components=5, max_iter='auto', random_state=97, method='fastica')
ica.fit(raw_highpass)
ica.plot_sources(raw, show_scrollbars=False)
print()
ica.plot_components()
print()
'''



biosemi_montage = mne.channels.make_standard_montage('biosemi64')
#biosemi_montage.plot(show_names=False)
n_channels = len(biosemi_montage.ch_names)
fake_info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=100.,
                            ch_types='eeg')
# print(biosemi_montage.ch_names,len(biosemi_montage.ch_names))

pre_cor = np.load('pre_cor.npy')
print(np.max(pre_cor),np.min(pre_cor))
data = pre_cor
fake_evoked = mne.EvokedArray(data, fake_info)
fake_evoked.set_montage(biosemi_montage)

fake_mask = np.zeros((n_channels,10), dtype=bool)
# lik = 3,4,8,9,14,19
# new_lik = 4,14
fake_mask[34,3]=1
fake_mask[34,4]=1
fake_mask[33,3]=1
fake_mask[34,4]=1
fake_mask[0,4]=1
fake_mask[1,4]=1

para = dict(marker='*', markerfacecolor='gold', markeredgecolor='gold',
        linewidth=0, markersize=12)
times = np.arange(0,0.05,0.01)
fig = fake_evoked.plot_topomap(np.array(times),time_unit='s',scalings=1e2,units=None, mask = fake_mask, mask_params=para,time_format='')
fig.savefig('topomap_preference.pdf')
print()


val_cor = np.load('val_cor.npy')
aro_cor = np.load('aro_cor.npy')

data = np.c_[val_cor,aro_cor]
print(np.max(data),np.min(data))
fake_evoked = mne.EvokedArray(data, fake_info)
fake_evoked.set_montage(biosemi_montage)
#fake_evoked.set_montage('standard_1020')
#fig, ax = plt.subplots(ncols=2, figsize=(8, 4), gridspec_kw=dict(top=0.9),sharex=True, sharey=True)

'''channel = ['AF8', 'Fp2', 'Fp1', 'AF7', 'O1']
true_info = mne.create_info(channel , 250 , ch_types = "eeg")
true_data=np.zeors(5,10)
idx=[34,33,0,1,26]
for i in range(5):
    true_data[i,:] = data[idx[i],:]
true_evoked = mne.EvokedArray(true_data, true_info)
true_evoked.set_montage(biosemi_montage)
'''
fake_mask = np.zeros((n_channels,10), dtype=bool)

# val = 0,10
# new_val = 5,6,10,11
# aro = 4,9,19,24
fake_mask[34,0]=1
fake_mask[0,0]=1
fake_mask[34,9]=1
fake_mask[33,9]=1
fake_mask[1,9]=1
fake_mask[26,9]=1

para = dict(marker='*', markerfacecolor='gold', markeredgecolor='gold',
        linewidth=0, markersize=12)

## axes
##mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, axes=ax[0],show=True, show_names=True)
                     #names=biosemi_montage.ch_names,)
#mne.viz.tight_layout()
times = np.arange(0,0.1,0.01)
fig = fake_evoked.plot_topomap(np.array(times),time_unit='s',ncols=5,nrows='auto',scalings=1e2,units=None,
                        mask = fake_mask, mask_params=para,time_format='')
fig.savefig('topomap_mood.pdf')
# np.array(times)
# add titles
#ax[0].set_title('MNE channel projection', fontweight='bold')
print()