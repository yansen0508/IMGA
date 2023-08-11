#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import sys
import os
import glob
import csv
import codecs
import datetime
import random
from psychopy import prefs
prefs.general['audioLib'] = ['pyo']
from psychopy import visual,event,core,gui

import numpy as np
import pickle
import pandas as pd
# from sklearn import preprocessing
import matplotlib.pyplot as plt
# import seaborn as sb
from options import Options
from solvers import create_solver
# from PIL import Image
import math
# import imageio
from scipy import stats,signal

subject_info = {u'Number':1, u'Age':20, u'name': u'f/m'}
dlg = gui.DlgFromDict(subject_info, title=u'REVCOR')
if dlg.OK:
    subject_number = subject_info[u'Number']
    subject_age = subject_info[u'Age']
    subject_sex = subject_info[u'name']    
else:
    core.quit() #the user hit cancel so exit
date = datetime.datetime.now()
time = core.Clock()
n_au = 16
cross_rate = 0.5
# how many AUs will be kept ?
keep_genes = 3
mutate_rate = 0.03
trial_count = 0
n_stims = 10
intern_count = 0
mental_rep=np.array(np.zeros(17))
corr = np.array([])
first_idx = 0
second_idx = 1
#norm=10
threshold=0.9
# iteration : generation
generation = 50
iteration = 0
exploitation_ratio = 5
n_blocks = 1
winner = ""
loser = ""
au2ix={'1':0, '2':1, '4':2, '5':3, '6':4, '7':5,
     '9':6, '10':7, '12':8, '14':9, '15':10, '17':11, 
     '20':12, '23':13, '25':14, '26':15,'happy':16}
au_label = pickle.load(open('aus_openface_new560.pkl','rb'),encoding='iso-8859-1')
img_path = os.path.dirname(__file__)+'/images/'
sound_path = os.path.dirname(__file__)+'/subject/'
output_folder = sound_path + 'Subj1' 

def generate_result_file(subject_number):
#AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r
    result_file = os.path.dirname(__file__)+'/csv/results_subj'+str(subject_number)+'_'+date.strftime('%y%m%d_%H.%M')+'.csv'        
    result_headers = ['subj','trial', 'name', 'age', 'date', 'image','AUs (AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r)',
                      'decision','reponse time']
    with open(result_file, 'w+') as file:
        writer = csv.writer(file)
        writer.writerow(result_headers)
    return result_file

def show_text_and_wait(file_name = None, message = None):
    event.clearEvents()
    if message is None:
        with codecs.open (file_name, "r", "utf-8") as file :
            message = file.read()
    text_object = visual.TextStim(win, text = message, color = 'black')
    text_object.height = 0.05
    text_object.draw()
    win.flip()
    while True :
        if len(event.getKeys()) > 0: 
            core.wait(0.2)
            break
        event.clearEvents()
        core.wait(0.2)
        text_object.draw()
        win.flip()

def update_trial_gui(): 
    play_instruction.draw()
    play_icon.draw()
    play_icon1.draw()
    response_instruction.draw()
    
    for response_label in response_labels: response_label.draw()
    for response_checkbox in response_checkboxes: response_checkbox.draw()
    win.flip()      

def enblock(x, n_stims):
    # generator to cut a list of stims into blocks of n_stims
    # returns all complete blocks
    for i in range(len(x)//n_stims):
        start = n_stims*i
        end = n_stims*(i+1)
        yield x[start:end]
    
def generate_trial_files(subject_number,n_blocks,n_stims,iteration,output_folder):
# generates n_block trial files per subject
# each block contains n_stim trials, randomized from folder which name is inferred from subject_number
# returns an array of n_block file names
    
    seed = time.getTime()
    random.seed(seed+iteration)
    stim_folder = output_folder
    print("stims folder :", stim_folder)
    sound_files = [os.path.basename(x) for x in glob.glob(stim_folder+"/*.jpg")]
    random.shuffle(sound_files)
    sound_files.remove('0_0.jpg')
    first_half = sound_files[:int(len(sound_files)/2)]
    second_half = sound_files[int(len(sound_files)/2):]
    print("files : ",sound_files,len(sound_files))

    block_count = 0
    trial_files = []
#    for block_stims in enblock(list(zip(first_half, second_half)),n_stims):
    for block_stims in enblock(list(zip(first_half, second_half)),n_stims):
        trial_file = 'csv/trials/trials_subj' + str(subject_number) + '_' + str(block_count) + '_' + date.strftime('%y%m%d_%H.%M')+'_'+str(iteration)+'.csv'
        print ("generate trial file "+trial_file)
        trial_files.append(trial_file)
        with open(trial_file, 'w+') as file :
            # each trial is stored as a row in a csv file, with format: 
            # write header
            writer = csv.writer(file)
            writer.writerow(["StimA","StimB"])

            # write each trial in block
            for trial_stims in block_stims:   
                writer.writerow(trial_stims)
        # break when enough blocks
        block_count += 1
        if block_count >= n_blocks:
            break
    return trial_files
def read_trials(trial_file): 
# read all trials in a block of trial, stored as a CSV trial file
    with open(trial_file) as fid :
        print("trial file: ",trial_file)
        reader = csv.reader(fid)
        print("trial header: ",next(reader))

        trials = list(reader)
        
    return trials[0::] #trim header
    
    
au2ix={'1':0, '2':1, '4':2, '5':3, '6':4, '7':5,
     '9':6, '10':7, '12':8, '14':9, '15':10, '17':11, 
     '20':12, '23':13, '25':14, '26':15,'happy':16}
"""
## Genetic algo
"""
def crossover(loser_winner):      # crossover for loser
#    loser_winner = np.array([np.zeros(16),np.ones(16)])
    cross_idx = np.empty((n_au,)).astype(np.bool)
    seed = time.getTime()
    random.seed(seed+iteration)
    for i in range(n_au):
        cross_idx[i] = True if np.random.rand() < cross_rate else False  # crossover index

    loser_winner[0, cross_idx] = loser_winner[1, cross_idx]  # assign winners genes to loser
    
    # reducing the activated genes

    idx=np.array([],dtype=int)
    if sum(loser_winner[0])>keep_genes*2:
        reduce=int((sum(loser_winner[0])-keep_genes*2)/2)
        print("#reduce genes",reduce)
        for j in range(n_au):    
            if loser_winner[0,j]==2:
                idx=np.append(idx,j)
        idx=list(idx)

        choice=random.sample(idx,reduce)

        for reduce_idx in choice:
            loser_winner[0,reduce_idx]=0

    return loser_winner


def mutate(loser_winner):         # mutation for loser
    mutation_idx = np.empty((n_au,)).astype(np.bool)
    seed = time.getTime()
    random.seed(seed+iteration)
    for i in range(n_au):
        mutation_idx[i] = True if np.random.rand() < mutate_rate else False  # mutation index
#    print("mutation: \n",mutation_idx)
    # flip values in mutation points (bascule)
    loser_winner[0, mutation_idx] = ~loser_winner[0, mutation_idx].astype(np.bool)
    for i in range(n_au):
        if loser_winner[0, i] == 1 :
            loser_winner[0, i] = 2 
    return loser_winner

def max3(win):
    m1=max(win)
    win1=np.copy(win)
    loc1=np.argmax(win1)
    win1[np.argmax(win1)] = np.min(win)
    m2=max(win1)
    loc2=np.argmax(win1)
    win2=np.copy(win1)
    win2[np.argmax(win2)] = np.min(win2)
    m3=max(win2)
    loc3=np.argmax(win2)

    location = np.array([loc1,loc2,loc3])
    return m1,m2,m3,location

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return t
def Dtanh(x):
    if (x==0):
        t= 1
    else:
        t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))/x
        
    return t

def psnr(d,m):
    mse=np.mean(d**2)
    if mse<=1.0e-10:
        return 100
    return 20*math.log10(m/math.sqrt(mse))

#stop=0
win = visual.Window([1366,768],fullscr=False,color="lightgray", units='norm')
screen_ratio = (float(win.size[1])/float(win.size[0]))
isi = .5
location=np.array([])
# trial gui
question = u'which face seems to be angrier?'

response_options = ['[g] Left','[h] Right']
response_keys = ['g', 'h']
label_size = 0.07
play_instruction = visual.TextStim(win, units='norm', text='There are two faces', color='red', height=label_size, pos=(0,0))
response_instruction = visual.TextStim(win, units='norm', text=question, color='black', height=label_size, pos=(0,-0.1), alignHoriz='center')
play_icon = visual.ImageStim(win, image=img_path+'play_on.png', units='norm', size = (1*screen_ratio,1), pos=(-0.5,0.5+0.25*label_size))
play_icon1 = visual.ImageStim(win, image=img_path+'play_off.png', units='norm', size = (1*screen_ratio,1), pos=(0.5,0.5+0.25*label_size))
response_labels = []
response_checkboxes = []
reponse_ypos = -0.2
reponse_xpos = -0.1
label_spacing = abs(-0.8 - reponse_ypos)/(len(response_options)+1)
for index, response_option in enumerate(response_options):
    y = reponse_ypos - label_spacing * index
    response_labels.append(visual.TextStim(win, units = 'norm', text=response_option, alignHoriz='left', height=label_size, color='black', pos=(reponse_xpos,y)))
    response_checkboxes.append(visual.ImageStim(win, image=img_path+'rb_off.png', size=(label_size*screen_ratio,label_size), units='norm', pos=(reponse_xpos-label_size, y-label_size*.05)))

# generate data files
result_file = generate_result_file(subject_number)
# experiment 
show_text_and_wait(file_name="intro.txt")  
show_text_and_wait(file_name="practice.txt")  
corr_au=np.array([])
pearson_vector=np.array([])
response_key=''
#trial_file = generate_trial_files(subject_number,n_blocks,n_stims,iteration)
winners = np.array([])
losers = np.array([])
population = np.array([])
#trial_file = generate_trial_files(subject_number,n_blocks,n_stims,iteration)
counter=0
times=0

t_score=np.zeros((generation,1))
cost=np.zeros((generation,1))
histogram=np.zeros((generation,17))
histogram_lose=np.zeros((generation,17))
histogram_all=np.zeros((generation,17))

diff=np.zeros((generation-1,17))
diff_lose=np.zeros((generation,17))
diff_all=np.zeros((generation-1,17))
diff_wl=np.zeros((generation,17))
snr=np.zeros((generation-1,1))
snr_lose=np.zeros((generation,1))
snr_all=np.zeros((generation-1,1))
snr_wl=np.zeros((generation,1))

corr_l=np.zeros((generation,1))
corr_wl=np.zeros((generation,1))
corr_w=np.zeros((generation,1))

reward=np.zeros((generation,1))
reward1=np.zeros((generation,1))
reward2=np.zeros((generation,1))

while (iteration < generation) :  
    data = {}
    if iteration == 0:
        data['0'] = np.array(
        [0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.00, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.00, 0.00, 0.00, 0.00])

    else:
        #data for yan mask
        data['0'] = np.array(
        [0.79, 0.03, 0.61, 0.16, 0.54, 0.64,
         0.00, 0.28, 0.65, 0.16, 0.02, 1.20,
         1.48, 0.00, 0.00, 0.00, 0.00])


    pkl_num = 1
    filelist = np.array(['0.bmp'])    
    trial_file = generate_trial_files(subject_number,n_blocks,n_stims,iteration,output_folder)
    # block_trials : a list of all pairs of trials
    block_trials = read_trials(trial_file[0])         
    # trial : the trial pair [*.jpg, *.jpg]
    for trial in block_trials :

        play_instruction.setColor('black')
        
        for checkbox in response_checkboxes:
            checkbox.setImage(img_path+'rb_off.png')
        
        seed = time.getTime()
        random.seed(seed+iteration)
        rand_num=random.randint(0,9)%2
        print("random num",rand_num)
        if rand_num==1 :
            sound_1 = output_folder +'/'+trial[0]
            print(trial[0])
            print("left AU: ",au_label[trial[0][2:-4]])
            sound_2 = output_folder +'/'+trial[1]
            print(trial[1])
            print("right AU: ",au_label[trial[1][2:-4]])

        else:
            sound_1 = output_folder +'/'+trial[1]
            print(trial[1])
            print("left AU: ",au_label[trial[1][2:-4]])
            sound_2 = output_folder +'/'+trial[0]
            print(trial[0])
            print("right AU: ",au_label[trial[0][2:-4]])     
            
        end_trial = False
        while (not end_trial):

            play_icon.setImage(sound_1)
            play_icon1.setImage(sound_2)
            # focus response instruction
            response_start = time.getTime()
            response_instruction.setColor('red')
            update_trial_gui()
            # upon key response...
            response_key = event.waitKeys(keyList=response_keys)
            response_time = time.getTime() - response_start
            # unfocus response_instruction, select checkbox
            response_instruction.setColor('black')
            response_checkboxes[response_keys.index(response_key[0])].setImage(img_path+'rb_on.png')
            update_trial_gui()
            # blank screen and end trial
#            core.wait(0.3) 
            win.flip()
#            core.wait(0.2) 
            end_trial = True
        
        # log response
        row = [subject_number, trial_count, subject_sex, subject_age, date]
        if response_key == ['g']:
            response_choice = 0
        elif response_key == ['h']:
            response_choice = 1
        print("response: ",response_choice)
        # write down the results        
        with open(result_file, 'a') as file :
            writer = csv.writer(file,lineterminator='\n')
        # index [2:-4], stim: file name ;  stime_order = true/false
            for stim_order,stim in enumerate(trial):
                print("stim_order , stim ",stim_order,stim)
#                print("block_trial  ",block_trials[intern_count][stim_order])
                # if it is a winner
                if (response_choice == stim_order):
                    mental_rep = mental_rep + au_label[stim[2:-4]]   
                    corr=np.append(corr,np.append(np.array(au_label[stim[2:-4]]),1))
                    if rand_num==1 :
                        winner = au_label[stim[2:-4]][0:16]# deactivate the last AU (eye close)
                        print("winner",winner)
                        winners= np.append(winners,au_label[stim[2:-4]])
                    else:
                        loser = au_label[stim[2:-4]][0:16]# deactivate the last AU (eye close)
                        print("loser",loser)
                        losers= np.append(losers,au_label[stim[2:-4]])
                # if it is a loser
                else:
                    mental_rep = mental_rep - au_label[stim[2:-4]]
                    corr=np.append(corr,np.append(np.array(au_label[stim[2:-4]]),0))
                    if rand_num==1 :
                        loser = au_label[stim[2:-4]][0:16]# deactivate the last AU (eye close)
                        print("loser",loser)
                        losers= np.append(losers,au_label[stim[2:-4]])
                    else:   
                        winner = au_label[stim[2:-4]][0:16]# deactivate the last AU (eye close)
                        print("winner",winner)
                        winners= np.append(winners,au_label[stim[2:-4]])
                result = row + [stim,au_label[stim[2:-4]],response_choice==stim_order,round(response_time,3)]
                
                writer.writerow(result)
            
        """
        # genetic algo
        # winer_loser ==> cross_over ==> mutaion ==> replace loser (generate a new trial, update pkl csv files of GAN)
        """
        # cross over, mutation
        loser_winner = np.array([loser,winner],dtype=int)# the first is loser
        loser_winner = crossover(loser_winner)
        loser_winner = mutate(loser_winner)
        data[str(pkl_num)] = np.append(loser_winner[0],0)
        data[str(pkl_num+1)] = np.append(loser_winner[1],0)
        filelist = np.append(filelist,str(pkl_num) + '.bmp')
        filelist = np.append(filelist,str(pkl_num+1) + '.bmp')
        intern_count += 1        
        trial_count += 1  
        pkl_num +=2
        print("It remains ",generation*n_stims - trial_count," trials")

    win_loc=np.reshape(winners,(-1,17))
    lose_loc=np.reshape(losers,(-1,17))
    hist=sum(win_loc[iteration*n_stims:iteration*n_stims+n_stims])
#    t=tanh(hist/200)
    hist_lose=sum(lose_loc[iteration*n_stims:iteration*n_stims+n_stims])
    histo_all=hist+hist_lose
    
    histogram_lose[iteration]=hist_lose
    histogram[iteration]=hist
    histogram_all[iteration]=histo_all
    
    diff_wl[iteration]=histogram[iteration]-histogram_lose[iteration]
    snr_wl[iteration]=psnr(diff_wl[iteration],max(histogram_all[iteration]))
    corr_wl[iteration]=stats.pearsonr(histogram_lose[iteration],histogram[iteration])[0]

    if iteration>0:        
   
        corr_l[iteration]=stats.pearsonr(histogram_lose[iteration],histogram_lose[iteration-1])[0]
        corr_w[iteration]=stats.pearsonr(histogram[iteration],histogram[iteration-1])[0]

    reward1[iteration]=0.5*corr_l[iteration]+0.5*corr_wl[iteration]
    print("reward: ",reward1)
    
    if reward1[iteration]>threshold:
        keep_genes+=1
        
        if keep_genes>5:
            keep_genes=5
        if keep_genes==5 and reward1[iteration]>0.95:
            iteration=generation
    
    if iteration< generation:
        # replace (remember to add the last AU into the array)
        output_folder = "datasets/results/test_ganimation_30"
        output = open("datasets/test/aus_openface.pkl", 'wb')
        pickle.dump(data, output)
        output.close()
        
        df = pd.DataFrame(filelist)
        df.to_csv('datasets/test/train_ids.csv',header=False,index=False)
        df.to_csv('datasets/test/test_ids.csv',header=False,index=False)
        au_label = pickle.load(open('datasets/test/aus_openface.pkl','rb'),encoding='iso-8859-1')
        # generate new trial
        opt = Options().parse()
        solver = create_solver(opt)
        solver.run_solver()
        print('END of GAN')
    
    
    iteration +=1
            
    
show_text_and_wait(file_name="end_practice.txt")           
#End of experiment
show_text_and_wait("end.txt")

winners=np.reshape(winners,(-1,17))
losers=np.reshape(losers,(-1,17))
np.savetxt("./csv/MGA_all_winners"+'_'+date.strftime('%y%m%d_%H.%M')+".csv",winners,delimiter=',')
np.savetxt("./csv/MGA_all_losers"+'_'+date.strftime('%y%m%d_%H.%M')+".csv",losers,delimiter=',')

fig=plt.figure()
plt.plot(corr_l,label='inter population losers')
plt.plot(corr_w,label='intra population winners')
plt.plot(corr_wl,label='intra population')
plt.plot(reward1,label='reward')
plt.grid()
plt.legend()

#corr_au_reshape=corr_au.reshape(-1,2)
#corr_au_reshape=corr_au_reshape.astype(int)
#np.savetxt("./csv/MGA_acc_AUs"+'_'+date.strftime('%y%m%d_%H.%M')+".csv",corr_au_reshape,delimiter=',')

#calculate time
df_time=pd.read_csv('./csv/results_subj'+str(subject_number)+'_'+date.strftime('%y%m%d_%H.%M')+'.csv')
df_time=df_time['reponse time']
df_time=df_time.ix[df_time<8]
print('total time :(minutes) ',df_time.sum()/60/2)
# Close Python
win.close()
core.quit()
sys.exit()


