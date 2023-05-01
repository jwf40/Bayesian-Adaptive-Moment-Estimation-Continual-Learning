import torch
import matplotlib.pyplot as plt
import pickle 
import os
import numpy as np
from math import ceil

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
path = 'results/test_acc/Labelled_Final/'
exp = "CIsplitmnist"
grad = "graduated_False"


# with open('results/test_acc/Final_Shuffle_NoLabels/GRADUATED_DRAW_PROBS_shuffle_True_run_0', 'rb') as f:
#     draw_probs = pickle.load(f)
#     for i, li in enumerate(zip(*draw_probs)):
        
#         plt.plot(li, label=str(i))
#     plt.show()

n_tasks = 5
test_every=400 if exp=='pmnist' else 40

with open(f'results/exp_bounds/{exp}_graduated_task_boundaries_shuffle_True_run_0','rb') as f:
    exp_params = np.array(pickle.load(f))
    print(exp_params)
avgs = {'VCL': 0.0, 'MAS': 0.0, 'ONLINE EWC': 0.0, 'BufferBAdam': 0.0, 'BGD': 0.0, 'SI': 0.0}
running_avgs = {'VCL': [], 'MAS': [], 'ONLINE EWC': [], 'BufferBAdam': [], 'BGD': [], 'SI': []}
for fi in os.listdir(path):
    if exp in fi and grad in fi:            
        with open(path+fi, 'rb')as f:
            dat = pickle.load(f)
            print('dat',dat)            
            avg_score = np.zeros(len(dat[0]))
            alg = fi.split('_')[0]
            if alg=='TFCL':
                continue
            #print(alg)
            fins = [lis[-1] for lis in dat]
            avgs[alg]+= np.mean(fins)
            for i in range(len(dat[0])):
                avg_score[i] = np.mean([lis[i] for lis in dat])#[:1+n_tasks_to_date]
                # for task_id, task in enumerate(dat):    
                #     if task_id<=n_tasks_to_date:
                #         avg_score[i] += task[i]
                # avg_score[i] /= (n_tasks_to_date+1)
            running_avgs[alg].append(avg_score)
        
print(running_avgs)
for key in sorted(running_avgs.keys()):
    if len(running_avgs[key]) > 0:
        err = np.std(running_avgs[key], axis=0)
        mean = np.mean(running_avgs[key], axis=0)
        #plt.errorbar(range(0, 40*len(mean), 40), mean, yerr=err, label=key)
        plt.plot(range(0, 5), mean, label=key, linewidth=2)

        

for each in avgs.keys():
    avgs[each] /= 10
print(avgs)


plt.vlines(x=np.arange(0,5,1), ymin=0.1, ymax=0.5, colors='black', linestyles='dashed',label='New Task Introduced', alpha=0.4)
plt.title(f'Average Accuracy Across All Tasks For Domain-Incremental PMNIST')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
print("REMEMBER TO CHECK BATCH PATH")

