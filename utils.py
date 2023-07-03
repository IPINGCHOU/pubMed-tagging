import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import collections

from tqdm.auto import tqdm
from torch.autograd import Variable
from transformers import AutoModel


class pubMedDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer ,max_length = 30, batch_size = 128):
        self.len = len(texts)
        self.bs = batch_size
        
        temp = []
        for i in tqdm(np.arange(0, len(texts), self.bs)):
            toks = tokenizer.batch_encode_plus(texts[i : i +self.bs],
                                               padding = "max_length",
                                               max_length = max_length,
                                               truncation = True,
                                               return_tensors = "pt")
            
            temp.append(toks)

        self.input_ids = [i["input_ids"] for i in temp]
        self.token_type_ids = [i["token_type_ids"] for i in temp]
        self.attention_masks = [i["attention_mask"] for i in temp]

        self.input_ids = torch.concat(self.input_ids, axis = 0)
        self.token_type_ids = torch.concat(self.token_type_ids, axis = 0)
        self.attention_masks = torch.concat(self.attention_masks, axis = 0)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_masks[index]


class SapBERTembeddedSOM(nn.Module):
    # SOM Reference : https://github.com/giannisnik/som/blob/master/som.py
    # SapBERT Reference: https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext

    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
    def __init__(self, m, n, dim, niter, batch_size = 128, alpha=None, sigma=None):
        super(SapBERTembeddedSOM, self).__init__()
        self.m = m
        self.n = n
        self.comps = m * n
        self.dim = dim
        self.niter = niter
        self.bs = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("==== Trained on: ", self.device)

        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)

        self.weights = torch.FloatTensor(m*n, dim).uniform_(-1, 1).to(self.device)
        self.locations = torch.LongTensor(np.array(list(self.neuron_locations())))
        self.pdist = nn.PairwiseDistance(p=2)

        self.sapModel = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(self.device)


    def get_weights(self):
        return self.weights

    def get_locations(self):
        return self.locations

    def neuron_locations(self):
        for i in range(self.m):
            for j in range(self.n):
                yield np.array([i, j])

    def map_vects(self, i, t, a):
        out = []
        # vectorized version
        embedded = self.get_embedded(i, t, a)
        bs = len(embedded)

        input1 = embedded.repeat(1, self.comps).reshape(bs, self.comps, self.dim)
        input2 = self.weights.repeat(bs, 1).reshape(bs, self.comps, self.dim)
        dists = self.pdist(input1, input2)
        _, bmu_index = torch.min(dists, 1)
        bmu_loc = torch.LongTensor(np.array(list(map(lambda i: self.locations[i, :], bmu_index))))
        out.extend(bmu_loc.cpu().detach().tolist())
        return out
    
    def give_density_map(self, inputLoader):
        mapped = []
        print("=====mapping")
        for i, data in tqdm(enumerate(inputLoader), total = len(inputLoader)):
            mapped.extend(self.map_vects(*data))
        
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
                    
    def save_model(self, iter):
        with open('model_{0}.pkl'.format(iter), 'wb') as f:
            pickle.dump(self.weights.cpu().detach().numpy(), f)

    def get_embedded(self, input_ids, token_type_ids, attention_masks):
        # output dim: (bs, 768)
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)

        cls_rep = self.sapModel(input_ids, token_type_ids, attention_masks)[0][:,0,:]
        cls_rep = cls_rep

        return cls_rep


    def forward(self, i, t, a, it):
        # vectorized version
        embedded = self.get_embedded(i, t, a)
        bs = len(embedded)

        input1 = embedded.repeat(1, self.comps).reshape(bs, self.comps, self.dim)
        input2 = self.weights.repeat(bs, 1).reshape(bs, self.comps, self.dim)

        dists = self.pdist(input1, input2)
        
        bmu_dist, bmu_index = torch.min(dists, 1)
        
        bmu_loc = torch.LongTensor(np.array(list(map(lambda i: self.locations[i, :], bmu_index))))
        bmu_loc_shape = bmu_loc.size()

        learning_rate_op = 1.0 - it/self.niter
        # learning_rate_op = 1 * np.power(0.5, it)
        alpha_op = self.alpha * learning_rate_op
        sigma_op = max(self.sigma * learning_rate_op, 1)
        
        bmu_distance_squares = torch.pow(self.locations.float().repeat(bs, 1).reshape(bs, self.comps, bmu_loc_shape[-1]) -
                                         bmu_loc.float().repeat(1, self.comps).reshape(bs, self.comps, bmu_loc_shape[-1]), 2).sum(-1)
        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op**2)))
        learning_rate_op = alpha_op * neighbourhood_func

        learning_rate_multiplier = learning_rate_op.repeat(1, self.dim).reshape(bs, self.comps, self.dim).to(self.device)
        delta = torch.div(torch.mul(learning_rate_multiplier, input1 - input2).sum(0), bs)                          
        self.weights = torch.add(self.weights, delta).detach()
        
        return bmu_dist.sum().cpu().detach().numpy()