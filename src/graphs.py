import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle 
fi = 'Results/pmnist/matchmaking_head_running_test_acc'

with open(fi,'rb') as f:
    data  = pickle.load(f)

fi2 = 'Results/pmnist/decentralised_matchmaking_head_running_acc'
with open(fi2,'rb') as f:
    data2  = pickle.load(f)
data = [n/900 for n in data]
data2 = [n/900 for n in data2]

sns.set_theme()
sns.color_palette('Set2')
plt.plot(data[:len(data2)-1],label='Centralised Test Accuracy')
plt.plot(data2,label='Decentraliesd Test Accuracy')
plt.legend(loc='lower right')
plt.title('Matchmaking Head Accuracy For 10 Agents')
plt.xlabel('Batches [128 samples/batch]')
plt.ylabel('Model Accuracy')
plt.show()