import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class BaseCLMethod:
    def __init__(self, model, train_loader, test_loader, file_name, **kwargs):
        self._run = kwargs['run']
        self._shuffle = kwargs['shuffle']
        self.model = model.to(kwargs["device"])
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.buffer_len = kwargs['buffer_len']
        self.test_every = kwargs['test_every']
        self.dim=next(iter(train_loader[0]))[0].shape[1]
        self.epochs = kwargs["epochs"]
        self.device = kwargs["device"]
        self.use_labels = kwargs["labels"]
        self.root = kwargs["root"]
        self.file_name = file_name
        self.optim = optim.Adam(
            self.model.parameters(), lr=0.001, betas=(0.9,0.999)
        )  # , momentum=0.9
        self.criterion = nn.CrossEntropyLoss()
        self.task_counter = 0
        self.test_acc_list = [[] for _ in range(len(self.test_loader))]
        self.kwargs = kwargs
        
    def save_draw_probs(self):
        probs = []
        if self.kwargs['graduated']:
            for data in self.train_loader[0]:
                probs.append(self.train_loader[0].draw_probs)
            self.save(probs, self.root+'GRADUATED_DRAW_PROBS')

    def get_task_boundaries(self, graduated=False):
        loaders = self.train_loader
        if graduated:
            loaders = self.train_loader[0].dloaders
        boundaries = []
        for idx, each in enumerate(loaders):
            if idx == 0:
                boundaries.append(len(each))
            else:
                boundaries.append(len(each)+ boundaries[idx-1])
        return boundaries


    def run(self):
        #self.save_draw_probs()
        for task in tqdm(self.train_loader):
            self.train(task)
            self.test()
        self.save(self.test_acc_list, self.root+self.file_name)
        is_graduated = 'graduated' if self.kwargs['graduated'] else 'with_boundaries'
        self.save(self.get_task_boundaries(self.kwargs['graduated']), f"results/exp_bounds/{self.kwargs['exp']}_{is_graduated}_task_boundaries")

    def train(self, loader):
        raise NotImplementedError

    def test(self):
        for idx, task in enumerate(self.test_loader):
            task_acc = 0
            for data in task:
                x, y = data[0].to(self.device), data[1].to(self.device)
                preds = self.model(x)
                task_acc += float((torch.argmax(preds, dim=1) == y).sum()) / len(y)            
            task_acc /= len(task)
            self.test_acc_list[idx].append(task_acc)
            #print(f"Task {idx} Accuracy: {task_acc}")

    def save(self, obj,file_path):
        with open(file_path+f'_shuffle_{self._shuffle}_run_{str(self._run)}', 'wb') as f:
            pickle.dump(obj, f)

    # def zerolike_params_dict(self):
    #     return dict([(n, torch.zeros_like(p))
    #             for n, p in self.model.named_parameters()])

    def zerolike_params_dict(self, model=None):
        if model == None:
            model = self.model
        return dict([(n, torch.zeros_like(p)) for n, p in model.named_parameters()])

    def copy_params_dict(self):
        return dict([(n, p.data.clone()) for n, p in self.model.named_parameters()])
