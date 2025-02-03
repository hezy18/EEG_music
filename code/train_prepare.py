from pickletools import read_unicodestringnl
import pandas as pd
import numpy as np
import json

uid2name = {1: 'mfu15', 2: 'mfu9', 3: 'mfu10', 4: 'mpu2', 5: 'mfu16', 6: 'mfu6', 7: 'mfu14', 8: 'mpu4', 9: 'mfu3', 10: 'mfu4', 11: 'mfu18', 12: 'mfu7', 13: 'mfu11', 14: 'mfu8', 15: 'mfu2', 16: 'mfu13', 17: 'mfu12', 18: 'mfu17', 19: 'mfu1'}
iid2item = {1: 'SOBNXDP12A6D4F8880', 2: 'SOJSHBN12A8C138AAC', 3: 'SOGPBAW12A6D4F9F22', 4: 'SOICNON12A8C140437', 5: 'SOAOAHZ12A8C13AAF1', 6: 'SONGOJV12AF729AEBD', 7: 'SOVKSNI12AB018A15B', 8: 'SOZYNNT12A81C22E0F', 9: 'SOPXXDD12AB0184A81', 10: 'SOXMZYS12AB018349B', 11: 'SOZVILY12AB01855ED', 12: 'SOVSELF12A670215EB', 13: 'SOITRTA12A6D4F8261', 14: 'SODLCIU12A8AE45F1D', 15: 'SOOTKWZ12AB0181082', 16: 'SOEBUPS12AF729F765', 17: 'SOWQQSX12A58A7B81D', 18: 'SOTWSMJ12A6D4FB9DF', 19: 'SOWDLPO12A6D4F72BB', 20: 'SOCYOTN12A8C13A4B1', 21: 'SOUHNQN12AF72A3DE3', 22: 'SOEWPQB12A8C13222B', 23: 'SOZDTTJ12AB0183DFE', 24: 'SORWJSF12A8C138AB6', 25: 'SOQAIAY12AB017B7AC'}

name2uid = {uid2name[uid]:uid for uid in list(uid2name.keys())}
item2iid = {iid2item[iid]:iid for iid in list(iid2item.keys())}

def make_user_item_dict():
    user_df = pd.read_csv('/work/hzy/EEG_music/cut_data/user.csv','\t',usecols=(1,2,3))
    uid_data_dict = {}
    for i in range(len(user_df)):
        uid = user_df['u_id_c'][i]
        if uid == 0:
            continue
        uid_data_dict[uid] = [user_df['u_age_i'][i],user_df['u_gender_c'][i]]
    
    
    series = np.load('/work/hzy/EEG_music/cut_data/embedding_serires.npy')
    item_df = pd.read_csv('/work/hzy/EEG_music/cut_data/item_mid.csv','\t')
    #print(len(item_df),series.shape[0])
    iid_data_dict = {}
    for i in range(len(item_df)):
        iid = item_df['i_id_c'][i]
        if iid == 0:
            continue
        # i_popularity_i	i_year_c	i_danceability_f	i_energy_f	i_loudness_f	i_speechiness_f	i_acousticness_f	i_instrumentalness_f	i_liveness_f	i_valence_f	i_tempo_f
        iid_data_dict[iid] = [item_df['i_popularity_i'][i],item_df['i_year_c'][i],item_df['i_danceability_f'][i],item_df['i_energy_f'][i],
                              item_df['i_loudness_f'][i],item_df['i_speechiness_f'][i],item_df['i_acousticness_f'][i],item_df['i_instrumentalness_f'][i],
                              item_df['i_liveness_f'][i],item_df['i_valence_f'][i],item_df['i_tempo_f'][i]]
        embed = list(series[i,:])
        iid_data_dict[iid].extend(embed)
    
    return uid_data_dict, iid_data_dict, 

uid_data_dict,iid_data_dict = make_user_item_dict()

label_df = pd.read_csv('/work/hzy/EEG_music/cut_data/label.csv','\t',usecols=(1,2,3,4,5,6,7))

pref={}
user_data={} # age, gender
item_data={} # all
for i in range(len(label_df)):
    path = label_df['path'][i]
    pref[path]=label_df['rating'][i]
    user = label_df['user'][i]
    user_data[path] = uid_data_dict[name2uid[user]]
    item = label_df['echo_id'][i]
    item_data[path] = iid_data_dict[item2iid[item]]
dim = len(user_data[path])+len(item_data[path])

with open('/work/hzy/EEG_music/EEG_analysis/psd_filt.json') as f:
    data_psd = json.load(f)

data = np.zeros((len(pref),dim))
data_withpsd = np.zeros((len(pref),dim+25))
data_withpsdall = np.zeros((len(pref),dim+5))
data_noitem = np.zeros((len(pref),len(user_data[path])+25))

label = np.zeros((len(pref)))
rating = np.zeros((len(pref)))
i=0
for path in pref:
    data[i,:] = np.array(user_data[path]+item_data[path])
    data_noitem[i,:] = np.append(np.array(user_data[path]),np.array(data_psd[path]))
    data_withpsd[i,:] = np.append(data[i,:],np.array(data_psd[path]))
    all = []
    for j in range(5):
        all.append(data_psd[path][j*5+4])
    data_withpsdall[i,:] = np.append(data[i,:],np.array(all))
    if pref[path]<3:
        label[i]=0
    elif pref[path]==3:
        label[i]=1
    else:
        label[i]=2
    rating[i]=pref[path]
    i+=1



#np.save('label.npy',label)
np.save('rating.npy',rating)
np.save('data.npy',data)
np.save('data_withpsd.npy',data_withpsd)
np.save('data_withpsdall.npy',data_withpsdall)
np.save('data_noitem.npy',data_noitem)









