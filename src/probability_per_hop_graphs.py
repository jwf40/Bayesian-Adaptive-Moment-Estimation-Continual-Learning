import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle 
sns.set_theme()
sns.color_palette('Set2')

data = [0.1*i for i in range(1,11)]
data2 = [0.3, 0.53, 0.71, 0.83, 0.92,0.97,0.99]
width = 0.35
plt.bar(np.arange(len(data))-width,data, width=width,label='Training Accuracy')
plt.bar(np.arange(len(data))+width,[1 for i in range(len(data))], width=width,label='Test Accuracy')
plt.legend(loc='lower right')
plt.xticks(np.arange(len(data)))
plt.title('Matchmaking Head Accuracy For 10 Agents')
plt.xlabel('')
plt.ylabel('Model Accuracy')
plt.show()