import torch
import matplotlib.pyplot as plt
import pickle 
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
path = 'results/test_acc/Final_Shuffle_NoLabels/'
exp = "CI"
grad = "graduated_True"

with open('results/exp_bounds/pmnist_graduated_task_boundaries','rb') as f:
    exp_params = np.array(pickle.load(f))


avgs = {}
for fi in os.listdir(path):
    if exp in fi and grad in fi and 'Buffer' in fi:       
        with open(path+fi, 'rb')as f:
            dat = pickle.load(f)
            avg_score = []
            name = fi.split('_')[:-1]
            name = ' '.join(name)
            print(name)
            fins = [lis[-1] for lis in dat]
            print(np.mean(fins))
            if name in avgs.keys():
                avgs[name] += np.mean(fins)
            else:
                avgs[name] = np.mean(fins)
for each in avgs.keys():
    avgs[each] /= 5
    print('Average: ', each,': ', avgs[each])
