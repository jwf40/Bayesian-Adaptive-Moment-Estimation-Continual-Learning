import torch
from pmnist import pmnist_exp
if __name__ == '__main__':
    device='cuda'if torch.cuda.is_available() else 'cpu'
    pmnist_exp(device=device)