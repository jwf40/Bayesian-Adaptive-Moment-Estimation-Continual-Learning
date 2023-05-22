"""
Adapted From:
|| https://github.com/kkelchte/task_free_continual_learning
"""
import numpy as np
import os, sys, time
import numpy.random as rn
import torch
from .base import BaseCLMethod
from tqdm import tqdm
import numbers
class Task_free_continual_learning(BaseCLMethod):

    def __init__(self,model,train_loader,test_loader,**kwargs):
        

        super().__init__(model, train_loader, test_loader,\
                         file_name = f"TFCL_ds_{kwargs['exp']}_graduated_{kwargs['graduated']}", **kwargs)
        # Save settings
        self.verbose=True
        self.ntasks=len(train_loader)
        self.gradient_steps=1#kwargs['epochs']
        self.loss_window_length=5
        self.loss_window_mean_threshold=0.2
        self.loss_window_variance_threshold=0.1
        self.MAS_weight=kwargs['tfcl_lambda']#0.5
        self.recent_buffer_size=20#kwargs['buffer_len']
        self.hard_buffer_size=30
        
        

    def run(self,use_hard_buffer=False,continual_learning=True):
        
        count_updates=0
    
        stime=time.time()
        losses=[]
        test_accs={i:[] for i in range(len(self.test_loader))}
        recent_buffer=[]
        hard_buffer=[]
        # loss dectection
        loss_window=[]
        loss_window_means=[]
        loss_window_variances=[]
        update_tags=[]
        new_peak_detected=True
        # MAS regularization: list of 3 weights vectors as there are 3 layers.
        star_variables=[]
        omegas=[] #initialize with 0 importance weights
        for t in tqdm(range(self.ntasks)):
            for s, data in enumerate(tqdm(self.train_loader[t])):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # plt.scatter(inputs[t][s][0],inputs[t][s][1],color='red' if labels[t][s]==0 else 'blue')
                # save experience in replaybuffer
                recent_buffer.append({'state':inputs,
                                     'trgt':labels})
                if len(recent_buffer) > self.recent_buffer_size:
                    del recent_buffer[0]
                msg = ''
                # Train model on replaybuffer when it is full:
                if len(recent_buffer) == self.recent_buffer_size:
                    msg='task: {0} step: {1}'.format(t,s)

                    # get frames new samples from recent buffer, construct a batch of few samples.
                    x=[_['state'] for _ in recent_buffer]
                    y=[_['trgt'] for _ in recent_buffer]
                    
                    if use_hard_buffer and len(hard_buffer) != 0:
                        xh=[_['state'] for _ in hard_buffer]
                        yh=[_['trgt'] for _ in hard_buffer]

                    # run few training iterations on the received batch.
                                            # train self.model
                    self.optim.zero_grad()
                    for gs in range(self.gradient_steps):
                        # evaluate the new batch                        
                        y_pred = self.model(torch.cat(x).reshape(-1, self.dim))
                        y_sup=torch.cat(y).reshape(-1)                        
                        recent_loss = self.criterion(y_pred,y_sup)
                        #print(np.argmax(y_pred[:10].detach().cpu().numpy(), axis=1), y_sup[:10].detach().cpu().numpy(), recent_loss)
                        total_loss = torch.sum(self.criterion(y_pred,y_sup))

                        if use_hard_buffer and len(hard_buffer) != 0:
                            # evaluate hard buffer
                            yh_pred = self.model(torch.cat(xh).reshape(-1,self.dim))
                            yh_sup= torch.cat(yh)
                            
                            hard_loss = self.criterion(yh_pred,yh_sup)
                            total_loss += torch.sum(self.criterion(yh_pred,yh_sup))
                        
                        # keep train loss for loss window
                        if gs==0: first_train_loss=total_loss.detach().cpu().numpy()
                        
                        # add MAS regularization to the training objective
                        if continual_learning and len(star_variables)!=0 and len(omegas)!=0:
                            for pindex, p in enumerate(self.model.parameters()):
                                #if batchsize = 1
                                if isinstance(omegas[pindex], numbers.Number):
                                    total_loss+=self.MAS_weight/2.*torch.sum(float(omegas[pindex])*(p-star_variables[pindex])**2)
                                else:
                                    total_loss+=self.MAS_weight/2.*torch.sum(torch.from_numpy(omegas[pindex]).type(torch.float32)*(p-star_variables[pindex])**2)
                        
                        losses.append(total_loss)
                        torch.sum(total_loss).backward()
                        self.optim.step()
                
                    # save training accuracy on total batch
                    if use_hard_buffer and len(hard_buffer) != 0:
                        xt=x+xh
                        yt=y+yh
                    else:
                        xt=x[:]
                        yt=y[:]
                    yt_pred = self.model(torch.cat(xt).reshape(-1,self.dim))
                    accuracy = np.mean(np.argmax(yt_pred.detach().cpu().numpy(),axis=1)==yt)
                    msg+=' recent loss: {0:0.3f}'.format(np.mean(recent_loss.detach().cpu().numpy()))
                    if use_hard_buffer and len(hard_buffer) != 0:
                        msg+=' hard loss: {0:0.3f}'.format(np.mean(hard_loss.detach().cpu().numpy()))
                    
                    
                    # add loss to loss_window and detect loss plateaus
                    loss_window.append(np.mean(first_train_loss))
                    if len(loss_window)>self.loss_window_length: del loss_window[0]
                    loss_window_mean=np.mean(loss_window)
                    loss_window_variance=np.var(loss_window)
                    #check the statistics of the current window
                    if not new_peak_detected and loss_window_mean > last_loss_window_mean+np.sqrt(last_loss_window_variance):
                        new_peak_detected=True  
                    #time for updating importance weights    
                    if continual_learning and loss_window_mean < self.loss_window_mean_threshold and loss_window_variance < self.loss_window_variance_threshold and new_peak_detected:
                        count_updates+=1
                        update_tags.append(0.01)
                        last_loss_window_mean=loss_window_mean
                        last_loss_window_variance=loss_window_variance
                        new_peak_detected=False
                        
                        # calculate importance weights and update star_variables
                        gradients=[0 for p in self.model.parameters()]
                        
                        # calculate imporatance based on each sample in the hardbuffer
                        for sx in [_['state'] for _ in hard_buffer]:
                            self.model.zero_grad()
                            y_pred=self.model(torch.cat(sx).reshape(-1,self.dim))
                            torch.norm(y_pred, 2, dim=1).backward()
                            for pindex, p in enumerate(self.model.parameters()):
                                g=p.grad.data.clone().detach().cpu().numpy()
                                gradients[pindex]+=np.abs(g)
                                
                        #update the running average of the importance weights        
                        omegas_old = omegas[:]
                        omegas=[]
                        star_variables=[]
                        for pindex, p in enumerate(self.model.parameters()):
                            if len(omegas_old) != 0:
                                omegas.append(1/count_updates*gradients[pindex]+(1-1/count_updates)*omegas_old[pindex])
                            else:
                                omegas.append(gradients[pindex])
                            star_variables.append(p.data.clone().detach())
                        
                    else:
                        update_tags.append(0)
                    loss_window_means.append(loss_window_mean)
                    loss_window_variances.append(loss_window_variance)

                    #update hard_buffer
                    if use_hard_buffer:                    
                        if len(hard_buffer) == 0:
                            loss=recent_loss.detach().cpu().numpy()
                        else:
                            loss=torch.cat((recent_loss, hard_loss))
                            loss=loss.detach().cpu().numpy()
                            
                        hard_buffer=[]
                        loss=np.mean(loss)
                        sorted_inputs=[np.asarray(lx) for _,lx in reversed(sorted(zip(loss.tolist(),xt),key= lambda f:f[0]))]
                        sorted_targets=[ly for _,ly in reversed(sorted(zip(loss.tolist(),yt),key= lambda f:f[0]))]
                            
                        for i in range(min(self.hard_buffer_size,len(sorted_inputs))):
                            hard_buffer.append({'state':sorted_inputs[i],
                                               'trgt':sorted_targets[i]})
                                        # empty recent buffer after training couple of times
                    recent_buffer = []
                #evaluate on test set
                if s%self.test_every==0:
                    self.test()
                    # for i in range(len(self.test_loader)):
                #         t_acc = 0.0
                #         for batch in self.test_loader[i]:
                #             xtest, ytest = batch[0].to(self.device), batch[1].to(self.device)
                #             y_pred=self.model(xtest).type(torch.float32)
                #             #loss=loss_fn(y_pred,y_sup).detach().numpy()
                #             t_acc+=(torch.sum(torch.argmax(y_pred,axis=1)==ytest)/len(ytest)).item()
                #             #print(np.argmax(y_pred.detach().cpu().numpy()[:10], axis=1),ytest[:10], torch.sum(torch.argmax(y_pred,axis=1)==ytest), len(ytest))
                #         t_acc /= len(self.test_loader[i])
                #         test_accs[i].append(t_acc)
                #         msg+=' test[{0}]: {1:0.3f}'.format(i,t_acc)
                # if self.verbose:
                #     print(msg)

            
        if False and use_hard_buffer:
            xs_pos=[_['state'][0] for _ in hard_buffer if _['trgt']==1]
            ys_pos=[_['state'][1] for _ in hard_buffer if _['trgt']==1]
            xs_neg=[_['state'][0] for _ in hard_buffer if _['trgt']==0]
            ys_neg=[_['state'][1] for _ in hard_buffer if _['trgt']==0]
            plt.scatter(xs_pos,ys_pos,color='blue')
            plt.scatter(xs_neg,ys_neg,color='red')
            plt.title('replay buffer')
            plt.show()
            
        if False:
            for q in range(self.ntasks):
                y_pred=model(torch.from_numpy(test_inputs[q].reshape(-1,self.dim)).type(torch.float32)).detach().numpy()
                positive_points=[test_inputs[q][i] for i in range(len(test_inputs[q])) if np.argmax(y_pred[i])==1]
                negative_points=[test_inputs[q][i] for i in range(len(test_inputs[q])) if np.argmax(y_pred[i])==0]
                plt.scatter([p[0] for p in positive_points],[p[1] for p in positive_points],color='blue')
                plt.scatter([p[0] for p in negative_points],[p[1] for p in negative_points],color='red')
            plt.axis('off')
            plt.show()

        self.test()        
        print("duration: {0}minutes, count updates: {1}".format((time.time()-stime)/60., count_updates))
        self.save(self.test_acc_list, self.root+self.file_name)
        return losses, loss_window_means, update_tags, loss_window_variances, test_accs