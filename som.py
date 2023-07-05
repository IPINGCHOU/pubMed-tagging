import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import collections

from tqdm.auto import tqdm
from torch.autograd import Variable
from transformers import AutoModel

class SOM(nn.Module):
    # SOM Reference : https://github.com/fcomitani/simpsom/tree/main/simpsom

    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
    def __init__(self, m, n, dim, niter, epochs, lr):
        super(SOM, self).__init__()
        self.m = m
        self.n = n
        self.comps = m * n
        self.dim = dim
        self.start_sigma = max(self.m, self.n) / 2
        self.start_learning_rate = lr 
        self.niter = niter
        self.epochs = epochs
        self.tau = epochs / np.log(self.start_sigma)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("==== Trained on: ", self.device)
        
        
        # init parameters
        self.all_weights = torch.randn(self.m, self.n, self.dim).to(self.device)        
        # self.numerator = torch.zeros(self.m, self.n, self.dim).to(self.device)
        # self.denominator = torch.zeros(self.m, self.n, 1)
        # map coordinate
        self.unravel_precomputed = list(np.unravel_index(np.arange(self.comps),
                                                    (self.m, self.n)))
        self.unravel_precomputed[0] = torch.from_numpy(self.unravel_precomputed[0]).to(self.device)
        self.unravel_precomputed[1] = torch.from_numpy(self.unravel_precomputed[1]).to(self.device)
        
        # mesh
        self.xx, self.yy = np.meshgrid(np.arange(
                self.m), np.arange(self.n))
        self.xx = torch.from_numpy(self.xx).to(self.device)
        self.yy = torch.from_numpy(self.yy).to(self.device)
    
        
    def forward(self, data, iter):
        self.bs = len(data)
        # update parameters
        self._update_sigma(iter)
        self._update_learning_rate(iter)
        
        data = data.to(self.device)
        
        dists = self.pdist2(data, self.all_weights)
        bmu_idx = torch.argmin(dists, dim = 1)
        wins = (
            self.unravel_precomputed[0][bmu_idx],
            self.unravel_precomputed[1][bmu_idx]
        )
        
        neighbor = self.neighborhood_caller(wins) * self.learning_rate
        
        denominator = torch.sum(neighbor, 0).unsqueeze(-1)
        
        neighbor = neighbor.reshape(self.bs, -1)
        neighbor = torch.mm(torch.transpose(neighbor, 0, 1), data)
        neighbor = neighbor.view(self.m, self.n, self.dim)
        
        self.all_weights = torch.where(denominator != 0, neighbor / denominator, self.all_weights).detach()
        
    def _update_sigma(self, n_iter: int) -> None:
        self.sigma = self.start_sigma * np.exp(-n_iter / self.tau)

    def _update_learning_rate(self, n_iter: int) -> None:
        self.learning_rate = self.start_learning_rate * \
                             np.exp(-n_iter / self.epochs)
    
    
    def pdist2(self, x, w):
        # euclidean distance
        
        x_sq = torch.pow(x, 2).sum(axis = 1, keepdim = True).repeat(1, self.comps)
        
        w_flat = w.view(-1, self.dim)
        w_flat_sq = torch.pow(w_flat, 2).sum(axis = 1, keepdim = True).repeat(1, self.bs)
        result = x_sq + torch.transpose(w_flat_sq, 0, 1)
        result = result - 2 * torch.mm(x, torch.transpose(w_flat, 0, 1))
        return torch.sqrt(result)
        
    def neighborhood_caller(self, center):
        # xx: x coordinates in the grid mesh
        # yy: y coordinates in the grid mesh
        # center: index of the center point along the xx yy grid
        
        d = 2 * self.sigma ** 2
        
        nx = self.xx.unsqueeze(0)
        ny = self.yy.unsqueeze(0)
        cx = self.yy[center][:, None, None]
        cy = self.xx[center][:, None, None]
        
        px = torch.exp(-torch.pow(nx - cx ,2) / d)
        py = torch.exp(-torch.pow(ny - cy, 2) / d)
        
        p = torch.mul(px, py)
        
        return torch.permute(p, (0, 2, 1))
    
    def give_density_map(self, loader):
        out = []
        
        for i, vec in tqdm(enumerate(loader), total = len(loader)):
            out.extend(self.find_bmu_ix(vec))
        
        mapped = list(map(lambda x: [x // self.m, x % self.m], out))
        tupled = list(map(tuple, mapped))
        counts = collections.Counter(tupled)
        C = np.zeros([self.m, self.n])
        for i in range(self.m):
            for j in range(self.n):
                C[i][j] = counts[(i,j)]
        
        plt.imshow(C, origin="lower", cmap='gray', interpolation='nearest')
        plt.colorbar()
        plt.show()
        
        return C
    
    def find_bmu_ix(self, vecs):
        dists = self.pairdist(vecs, self.all_weights.reshape(self.m * self.n, self.dim))
        return np.argmin(dists, axis = 1)
    
    def pairdist(self, a,b):
        # euclidean
        a = a.to(self.device)
        a_sq = torch.sum(torch.pow(a, 2), dim = 1, keepdim = True)
        b_sq = torch.sum(torch.pow(b, 2), dim = 1, keepdim = True)
        b_sq_T = torch.transpose(b_sq, 0, 1)
        
        return torch.sqrt(a_sq + b_sq_T - 2 * torch.mm(a, torch.transpose(b, 0, 1))).cpu().detach().numpy()
    
    