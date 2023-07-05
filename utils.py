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

# ================ embedding only
class SapBERTembedded(nn.Module):
    # SOM Reference : https://github.com/giannisnik/som/blob/master/som.py
    # BatchSOM Reference : https://github.com/meder411/som-pytorch/blob/master/som.py
    # SapBERT Reference: https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext

    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
    def __init__(self):
        super(SapBERTembedded, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sapModel = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(self.device)
 
    def get_embedded(self, input_ids, token_type_ids, attention_masks):
        # output dim: (bs, 768)
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)

        cls_rep = self.sapModel(input_ids, token_type_ids, attention_masks)[0][:,0,:]
        cls_rep = cls_rep

        return cls_rep


    def forward(self, i, t, a):
        # batched version
        embedded = self.get_embedded(i, t, a)
        embedded = embedded.cpu().detach().numpy()
        return embedded




class SapBERTembeddedSOM(nn.Module):
    # SOM Reference : https://github.com/giannisnik/som/blob/master/som.py
    # BatchSOM Reference : https://github.com/meder411/som-pytorch/blob/master/som.py#L292
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
        
        self.contents = None
        self.grid = None
        self.grid_used = None
        self.grid_dists = None
        
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

        self.pdist = nn.PairwiseDistance(p = 2)
        self.sapModel = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(self.device)
        self.init_grid()

    def init_grid(self):
        self.contents = give_grid(self.m, self.n, self.dim).to(self.device)
        
        # create grid index matrix
        x, y = np.meshgrid(range(self.m), range(self.n))
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        self.grid = torch.stack((x,y)).to(self.device)
        self.grid_used = torch.zeros(self.m, self.n).long().to(self.device)
        
        self.grid_dists = pdist2(
            self.grid.float().view(2, -1),
            self.grid.float().view(2, -1),
            0
        ).to(self.device)

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
        min_index = self.find_bmu(embedded)
        out.extend(min_index.cpu().detach().tolist())
        return out
    
    def give_density_map(self, inputLoader):
        mapped = []
        print("=====mapping")
        for i, data in tqdm(enumerate(inputLoader), total = len(inputLoader)):
            mapped.extend(self.map_vects(*data))
        
        mapped = list(map(lambda x: [x // self.m, x % self.m], mapped))
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
            pickle.dump(self.contents.cpu().detach().numpy(), f)

    def get_embedded(self, input_ids, token_type_ids, attention_masks):
        # output dim: (bs, 768)
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)

        cls_rep = self.sapModel(input_ids, token_type_ids, attention_masks)[0][:,0,:]
        cls_rep = cls_rep

        return cls_rep

    def forward(self, i, t, a, it, sigma):
        # batched version
        
        # # parameters
        # learning_rate_op = 1.0 - it/self.niter
        # # learning_rate_op = 1 * np.power(0.5, it)
        # alpha_op = self.alpha * learning_rate_op
        # sigma_op = max(self.sigma * learning_rate_op, 1)
        
        # get embedding (input)
        embedded = self.get_embedded(i, t, a)
        
        # get weight
        weights = self.compute_weight(sigma, True)
        
        # find bmu
        min_index = self.find_bmu(embedded)
        
        # compute the freq with nodes are BMU
        freq_data = torch.zeros(self.comps).to(self.device)
        freq_data.index_add_(0, min_index, torch.ones(embedded.shape[0]).to(self.device))
        
        # store the updated freq fro each node
        self.grid_used += (freq_data != 0).view(self.m, self.n).long()
        
        # compute aggregate data values for each neighborhood
        avg_data = torch.zeros(self.comps, self.dim).to(self.device)
        avg_data.index_add_(0, min_index, embedded)
        avg_data = avg_data / freq_data.view(-1, 1)
        
        # weighted neibourhood
        freq_weights = weights * freq_data.view(-1, 1)
        
        # fill the non-bmu nodes with existed contents
        unused_idx = (freq_data == 0).nonzero()
        if unused_idx.shape:
             avg_data[unused_idx, :] = self.contents.view(-1, self.dim)[unused_idx, :]
             
        # update
        update_num = (freq_weights.unsqueeze(2) * avg_data).sum(1)
        update_denom = freq_weights.sum(1)
        update = update_num / update_denom.unsqueeze(1)
        # idx to update
        update_idx = update_denom.nonzero()
        
        # copy old contents for magnitude computation
        old_c = self.contents.clone()
        
        # update the contents
        self.contents.view(-1, self.dim)[update_idx, :] = update[update_idx, :].detach()
        
        return torch.norm(self.contents - old_c, 2, -1).sum().cpu().detach().numpy()
    
    def compute_weight(self, sigma, weighted):
        if weighted:
            return torch.exp(-self.grid_dists / (2 * sigma**2))
        else:
            return (self.grid_dists < sigma).float()
        
    def find_bmu(self, x, k = 1):
        N = x.shape[0]
        diff = x.view(-1, 1, self.dim) - self.contents.view(1, -1, self.dim)
        dist = (diff ** 2).sum(-1)
    
        _, min_idx = dist.topk(k = k, dim = 1, largest = False)
        
        return min_idx.squeeze()
    
# ================ utils
def give_grid(m, n, dim):
    grid = torch.randn(m, n, dim)
    
    return grid

def pdist2(X, Y, dim):
	N = X.shape[abs(dim-1)]	
	K = Y.shape[abs(dim-1)]
	XX = (X ** 2).sum(dim).unsqueeze(abs(0-dim)).expand(K, -1)
	YY = (Y ** 2).sum(dim).unsqueeze(abs(1-dim)).expand(-1, N)
	if dim == 0:
		XX = XX.expand(K, -1)
		YY = YY.expand(-1, N)
		prod = torch.mm(X.transpose(0,1), Y)
	else:
		XX = XX.expand(-1, K)
		YY = YY.expand(N, -1)
		prod = torch.mm(X, Y.transpose(0,1))
	return XX + YY - 2 * prod


