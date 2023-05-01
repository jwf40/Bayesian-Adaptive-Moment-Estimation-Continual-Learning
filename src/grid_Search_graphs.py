import torch
import matplotlib.pyplot as plt
import pickle 
import os
import numpy as np
from math import ceil

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
path = 'results/test_acc/BADAM/'
exp = "pmnist"
grad = "graduated_True"

n_tasks = 5
test_every=400 if exp=='pmnist' else 40

with open(f'results/exp_bounds/{exp}_graduated_task_boundaries_shuffle_True_run_0','rb') as f:
    exp_params = np.array(pickle.load(f))
    print(exp_params)
avgs = {}
running_avgs = {}
for fi in os.listdir(path):
    if exp in fi and grad in fi and 'Buffer' in fi:       
        with open(path+fi, 'rb')as f:
            dat = pickle.load(f)
            print(fi)
            avg_score = np.zeros(len(dat[0]))
            alg = fi.split('_')[0]
            name = fi.split('_')[:-1]
            name = ' '.join(name)
            #print(alg)
            fins = [lis[-1] for lis in dat]
            if name in avgs.keys():
                avgs[name] += np.mean(fins)
            else:
                avgs[name] = np.mean(fins)
            for i in range(len(dat[0])):
                #there are 479 samples per task
                try:
                    n_tasks_to_date = min(np.where(exp_params>=(i*test_every))[0])
                except:
                    n_tasks_to_date = n_tasks
                avg_score[i] = np.mean([lis[i] for lis in dat])
                # for task_id, task in enumerate(dat):    
                #     if task_id<=n_tasks_to_date:
                #         avg_score[i] += task[i]
                # avg_score[i] /= (n_tasks_to_date+1)
            if name in running_avgs.keys():
                running_avgs[name].append(avg_score)
            else:
                running_avgs[name] = [avg_score]                

for key in sorted(running_avgs.keys()):
    err = np.std(running_avgs[key], axis=0)
    mean = np.mean(running_avgs[key], axis=0)
    #plt.errorbar(range(0, 40*len(mean), 40), mean, yerr=err, label=key)
    if mean[-1] > 0.75:
        plt.plot(range(0, test_every*len(mean), test_every), mean, label=key)

        

for each in avgs.keys():
    avgs[each] /= 3
print(avgs)



plt.vlines(x=exp_params, ymin=0.1, ymax=1.0, colors='black', linestyles='dashed',label='New Task Introduced', alpha=0.4)
plt.title(f'Model Average Accuracy Across All Tasks')
plt.xlabel('Task Introduction')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
print("REMEMBER TO CHECK BATCH PATH")

