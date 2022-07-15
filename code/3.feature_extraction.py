import numpy as np

# psd
def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    from mne.time_frequency import psd_array_multitaper

    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

cut_length=3000
import os
root = '/work/hzy/EEG_music/filt1_data'
EEG_feature = {}
for filename in os.listdir(root):
    if filename.endswith('.csv') or filename.endswith('.jpg'):
        continue
    for file in os.listdir(root+'/'+filename):
        if not file.endswith('.npy'):
            continue
        print(file)
        data = np.load(root + '/' +filename +'/' + file) # cut_length*5, 5
        psd_num = []
        for i in range(5):
            for band in [[4,8],[8,12],[12,30],[30,45],[0,125]]:
                psd_num.append(bandpower(data[:,i], 250, band, 'multitaper'))
        EEG_feature[file]=psd_num

import json
with open('EEG_analysis/psd_filt.json','w') as f:
    json.dump(EEG_feature,f)



