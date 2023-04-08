import torch
import matplotlib.pyplot as plt
import pickle 
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
path = 'results/test_acc/'
exp = "CIsplit"
grad = "graduated_True"

with open('results/exp_bounds/pmnist_graduated_task_boundaries','rb') as f:
    exp_params = np.array(pickle.load(f))
avgs = {'VCL': 0.0, 'TFCL': 0.0, 'MAS': 0.0, 'ONLINE EWC': 0.0, 'BAdam': 0.0, 'BGD': 0.0, 'SI': 0.0}
for fi in os.listdir(path):
    if exp in fi and grad in fi:            
        with open(path+fi, 'rb')as f:
            dat = pickle.load(f)
            avg_score = []
            alg = fi.split('_')[0]
            print(alg)
            if 'VCL' in fi:
                fins = [lis[-1] for lis in dat]
            elif 'TFCL' in fi:
                fins = [lis[-1] for lis in dat.values()]
            else:
                fins = [lis[-1].detach().cpu().item() for lis in dat]
            print(np.mean(fins))
            avgs[alg]+= np.mean(fins)
            continue
            for idx in range(0, len(dat[0]), 100):
                ele_score = 0
                try:
                    n_tasks = 1+max(np.where((idx*100) < exp_params)[0])
                except:
                    n_tasks = len(dat)
                for task in range(n_tasks):
                    ele_score+=(dat[task][idx]).detach().cpu().item()
                avg_score.append(ele_score/n_tasks)
            label = fi.split('_')[0]
            plt.plot(range(0, 100*len(dat[0]), 100), avg_score, label=label)

for each in avgs.keys():
    avgs[each] /= 10
print(avgs)

plt.vlines(x=exp_params, ymin=0.1, ymax=1.0, colors='black', linestyles='dashed',label='New Task Introduced', alpha=0.4)
plt.title(f'Model Average Accuracy Across All Tasks')
plt.xlabel('Task Introduction')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

