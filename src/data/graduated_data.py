import torch
import numpy as np
import random
import math
from typing import Union
from copy import deepcopy

class GraduatedDataLoader:
    def __init__(self, dloaders: list, k=12):
        self.dloaders = dloaders
        self.iter_loaders = [iter(dl) for dl in dloaders]
        self.n_tasks = len(dloaders)
        self.task_li = list(range(self.n_tasks))
        self.finished_tasks = []
        self.k = k
        self.len = sum(len(d) for d in dloaders)
        self._choices = [0 for _ in self.task_li]
        self.iter_num = 0
        self._init_task_peaks()
        self.update_draw_probs()    

    def _init_task_peaks(self):
        """
        Calculate the iteration where each task should have its peak probability.
        This is defined as the middle point of the task
        """
        self.task_peaks = []
        last_dlen = 0
        for idx, dloader in enumerate(self.dloaders):
            self.task_peaks.append(last_dlen + (len(dloader)//2))
            last_dlen += len(dloader)
            
        
    
    def update_draw_probs(self):
        """
        Probs come from squared distance from each task's median index
        Squared to make it exponentially less likely to choose distant tasks -> enforces non-i.i.d
        """
        distances = [abs(self.task_peaks[idx]-self.iter_num) for 
            idx in self.task_li]
        max_dist = max(distances)
        weights = [((max_dist-w)**self.k) for w in distances]
        total_weights = sum(weights)
        self.draw_probs = [w/total_weights for w in weights]
        
    def __len__(self):
        return self.len
      
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iter_num % 10 == 0:
            print(f"Task Choices: {self._choices}, Draw Probs: {self.draw_probs}")

        def _get_task_choice():
            for tsk in self.finished_tasks:
                self.draw_probs[tsk] = 0.0
            
            # allow for floating point error
            if sum(self.draw_probs) > 1e-5:
                task_choice = random.choices(self.task_li, weights=self.draw_probs, k=1)[0]
                return task_choice
            else:
                raise StopIteration



        #Select which task to learn from
        task_choice = _get_task_choice()

        if isinstance(task_choice, type(StopIteration)):
            return task_choice
        
        try:
            return_data = next(self.iter_loaders[task_choice])
            self._choices[task_choice] +=1
            self.iter_num += 1
            self.update_draw_probs()                
            return return_data
        except StopIteration:
            print(f"Finished Task {task_choice}")
            self.finished_tasks.append(task_choice)
            if len(self.finished_tasks) == len(self.task_li) or sum(self.draw_probs) < 1e-5:
                raise StopIteration
            else:
                return self.__next__()            



                
        
