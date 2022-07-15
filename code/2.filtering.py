'''import mne
import numpy as np
import matplotlib
file = 'mfu1/mfu1_11.npy'
data = np.load(file)
signal = data.T
channel = ['AF8', 'Fp2', 'Fp1', 'AF7', 'O1']
info = mne.create_info(channel , 250 , ch_types = "eeg")
# raw就是得到的结果了
raw = mne.io.RawArray(signal, info)
raw.plot(duration='60')'''

import matplotlib.pyplot as plt
import numpy as np
channel = ['AF8', 'Fp2', 'Fp1', 'AF7', 'O1']
def draw_eeg(data,file):
    from matplotlib.pyplot import MultipleLocator
    x = range(data.shape[0])
    plt.figure(figsize=(8,4))
    for i in range(5):
        plt.subplot(5,1,i+1)
        plt.plot(x, data[:,i],label=channel[i])
        plt.yticks([])
        #plt.ylabel(channel[i],fontsize=16)
        #ax = plt.gca()
        #ax.yaxis.set_label_position("right")
    
        # plt.ylim(0,1)
    
        new_x = [i*0.004 for i in x]
        plt.xticks(x,new_x) #0.002
        plt.tick_params(labelsize=16) 
        ax_all = plt.gca()

        ax_all.xaxis.set_major_locator(MultipleLocator(2500))



    plt.xlabel('time/s',fontsize=16)
    plt.suptitle('signal of '+file)
    if not os.path.exists('filt1_data'):
            os.makedirs('filt1_data')
    plt.savefig('filt1_data/signal of '+file+'.pdf')
    plt.show()


import os
import mne
root = '/work/hzy/EEG_music/cut_data'

is_do = False
for filename in os.listdir(root):
    if filename.endswith('.csv'):
        continue
    if filename == 'embedding_serires.npy':
        continue
    for file in os.listdir(root+'/'+filename):
        if not file.endswith('.npy'):
            continue
        if file == 'mfu7_15.npy':
            is_do = True
        if is_do == False:
            continue
        print(file)
        data = np.load(root + '/' +filename +'/' + file)
        draw_eeg(data,file+'_raw')

        signal = data.T
        channel = ['AF8', 'Fp2', 'Fp1', 'AF7', 'O1']
        info = mne.create_info(channel , 250 , ch_types = "eeg")
        raw = mne.io.RawArray(signal, info)
        raw_highpass = raw.copy().filter(l_freq=1, h_freq=None)
        data_highpass = raw_highpass.get_data()
        data = data_highpass.T
        draw_eeg(data,file+'_filt')
        
        if not os.path.exists('filt1_data'):
            os.makedirs('filt1_data')
        if not os.path.exists('filt1_data/'+filename):
            os.makedirs('filt1_data/'+filename)
        
        filepath = 'filt1_data/'+filename+'/'+file
        np.save(filepath,data,allow_pickle=True)

