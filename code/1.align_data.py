import time
from datetime import datetime
import os
import glob
import copy
import json

names=['mfu1','mfu2','mfu3','mfu4','mfu6','mfu7','mfu8','mfu9','mfu10','mfu11','mfu12','mfu13','mfu14','mfu15','mfu16','mfu17','mfu18','mpu2','mpu4']
# names=['mfu1']

count={}
def get_data(name): # 提取并保存能够使用的data：pre_data
    time_dict = {}  # music_id:{'is_pause':True/False, 'time':[[play_timestamp,stop_timestamp]]}
    
    time_path = 'EEG_labstudy_data/timestamps/' + name
    f = open(time_path+ '_audiotime.txt', encoding='utf-8')
    audio_lines = f.readlines()
    f.close()
    f = open(time_path+ '_pagetime.txt', encoding='utf-8')
    page_lines = f.readlines()
    f.close()

    start_time={} # {page_id: timestamp}
    for line in page_lines:
        op, id, ts = line.split('--')  # start--1--1626150150627
        if id not in time_dict:
            start_time[id] = ts.split('\n')[0]
        
    tmp_list = []
    tmp_play = False
    last_id = ''
    last_ts = str(int(time.time()*1000))
    num_lines = len(audio_lines)
    i = 0
    for line in audio_lines:
        op, id, ts, sc = line.split('--')  # play--1--1626150155124--0
        i+=1
        if id not in time_dict:
            time_dict[id] = {'is_pause':True, 'time':[]}
        if op == 'play':
            if tmp_play == False: # 新一轮play
                tmp_play = True
                tmp_list.clear()
                tmp_list.append(ts)
            else: # 上一轮play没结束又看到了play
                last_pause_ts = start_time[id] # 这一轮play的start
                tmp_list.append(last_pause_ts)
                time_dict[last_id]['is_pause'] = False
                time_dict[last_id]['time'].append(copy.deepcopy(tmp_list))
                # 然后处理这一轮play
                tmp_play = True
                tmp_list.clear()
                tmp_list.append(ts)
            if i == num_lines: # 最后一行
                tmp_list.append(last_ts)
                time_dict[id]['is_pause'] = False
                time_dict[id]['time'].append(copy.deepcopy(tmp_list))
        elif float(sc.split('\n')[0]) < 10:
            tmp_play = False
            last_id = id
            continue
        else:
            tmp_play = False
            tmp_list.append(ts)
            time_dict[id]['time'].append(copy.deepcopy(tmp_list))
        last_id = id

    pre_data = {}  # {music_id:{'is_pause':True, 'time'':{timestamp:[eeg1,eeg2,eeg3,eeg4,eeg5,...(一共*5)]}}
    ospath = 'EEG_labstudy_data/EEG_data'
    eeg_path = glob.glob(ospath + '/*' + name + '.txt')  # ['EEG_labstudy/EEG_data/20210713-130431-mfu1.txt']
    day = eeg_path[0].split('/')[2].split('-')[0]
    f = open(eeg_path[0], encoding='utf-8')
    lines = f.readlines()
    f.close()

    key_id = 0
    for line in lines:
        time_old = line.split(',')[0]  # 12-52-05.563192
        timestr = day + ' ' + time_old  # 20210713 12-52-05.563192'
        datetime_obj = datetime.strptime(timestr, "%Y%m%d %H-%M-%S.%f")
        obj_stamp = str(int(time.mktime(
            datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0))  # 1626151925563 (13位）
        while key_id < len(time_dict):
            key = list(time_dict.keys())[key_id]
            break_flag = False
            is_pause = time_dict[key]['is_pause']
            for play_ts, pause_ts in time_dict[key]['time']:
                if obj_stamp < play_ts:
                    break_flag = True
                    break
                if obj_stamp > pause_ts:
                    continue
                if key not in pre_data.keys():
                    pre_data[key] = {'is_pause':is_pause,'time':{}}
                pre_data[key]['time'][obj_stamp] = line.split('\n')[0].split(',')
                pre_data[key]['time'][obj_stamp].pop(0)
                for i in range(6):
                    pre_data[key]['time'][obj_stamp].pop(25)

                break_flag = True
                break
            if break_flag:
                break
            key_id += 1

    print(name, 'done')
    count[name]={}
    for key in pre_data.keys():
        count[name][key]=len(pre_data[key]['time'])
        print(key,pre_data[key]['is_pause'],count[name][key])

    if not os.path.exists('pre_data'):
        os.makedirs('pre_data')
    json_save = json.dumps(pre_data)
    f = open('pre_data/'+name+'_data.json', 'w')
    f.write(json_save)
    f.close()
    print('saved %s\n' % name)


for name in names:
    get_data(name)
json_save = json.dumps(count)
f = open('pre_data/count.json', 'w')
f.write(json_save)
f.close()


import numpy as np
# cut data and 被试内归一化
cut_length = 3000
def cut_data(name, normalization='subject'):
    f = open('pre_data/'+name+'_data.json','r')
    pre_data = json.load(f) # {music_id:{'is_pause':True, 'time':{timestamp:[eeg1,eeg2,eeg3,eeg4,eeg5,...(一共*5)]}}
    f.close()
    music_data = {}
    for music_id in pre_data.keys():
        length = len(pre_data[music_id]['time'])
        if length < cut_length:
            continue
        case = np.zeros((cut_length*5,5))
        j = 0
        for ts in pre_data[music_id]['time'].keys():
            if j == cut_length:
                break
            case_line = np.array(pre_data[music_id]['time'][ts]) # 25
            ar_re = case_line.reshape(5,5)
            case[j*5:j*5+5,:] = ar_re
            j+=1
        music_data[music_id] = case
    num_music = len(music_data.keys()) 
    all_data = np.zeros((cut_length*5, 5, num_music))
    
    for i in range(num_music):
        music_id = list(music_data.keys())[i]
        all_data[:,:,i] = music_data[music_id]

    if normalization=='subject': # 同一导联内，该被试所有case统一归一
        for i in range(5):
            need_nor_data = all_data[:,i,:]
            max = need_nor_data.max()
            min = need_nor_data.min()
            norred_data = (need_nor_data-min)/(max-min)#(0,1之间)
            all_data[:,i,:] = norred_data

    for k in range(num_music):
        music_id = list(music_data.keys())[k]
        music_data[music_id] = all_data[:,:,i]

    if not os.path.exists('cut_data'):
        os.makedirs('cut_data')
    if not os.path.exists('cut_data/'+name):
        os.makedirs('cut_data/'+name)
    for k in range(num_music):
        music_id = list(music_data.keys())[k]
        file_name = 'cut_data/'+name+'/'+name+'_'+music_id+'.npy'
        data = music_data[music_id]
        np.save(file_name,data,allow_pickle=True)


for name in names:
    cut_data(name)