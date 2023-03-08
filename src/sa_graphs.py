import torch
import matplotlib.pyplot as plt
import pickle 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

file_strs = list(filter(
        lambda x: 'single_agent' in x and '', os.listdir('Results/')
    ))

file_strs = ['Mean_Eta_0.1_Fast_True_bgd_single_agent','Mean_Eta_1_Fast_False_bgd_single_agent','Adam_True_sgd_single_agent']

#files = {"Fast_False_bgd_single_agent_COST":[], "Fast_True_bgd_single_agent_COST":[], "Adam_False_sgd_single_agent_COST":[]}
# files = {}
# for fi in file_strs:
#     with open('Results/'+fi, 'rb')as f:
#         files[fi] = pickle.load(f)
#         print(fi)
#         print(files[fi])
#         print("\n\n\n\n\n#################################\n")
#         plt.plot([i for idx,i in enumerate(files[fi]) if idx % 20 == 0], label=fi)
for fi in file_strs:
    with open('Results/'+fi, 'rb')as f:
        dat = pickle.load(f)
        label = 'Badam' if 'True' in fi else 'BGD'
        plt.plot(dat, label=label)

plt.vlines(x=[i for i in range(10)], ymin=0.1, ymax=1.0, colors='black', linestyles='dashed',label='New Task Introduced', alpha=0.4)
plt.title(f'Model Average Accuracy Across All Tasks')
plt.xlabel('Task Introduction')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
