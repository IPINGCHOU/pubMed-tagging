#%% import packages
import utils
import numpy as np
import pandas as pd
import torch
import re
import os
import matplotlib.pyplot as plt
import pickle
import random

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from collections import OrderedDict
#%% read files
# df = pd.read_excel("input_file.xlsx", sheet_name = "Sheet1", header = 0)

# #%% title - check and preprocess
# title = df["title"].to_list()
# title = list(map(lambda x: x.split(" "), title))

# title_length = [len(i) for i in title]
# plt.hist(title_length)
# title_length = np.array(title_length)
# for i in range(10, 45, 5):
#     temp = title_length < i
#     print("Length: {0}, covered: {1} %".format(i, sum(temp) / len(temp) * 100))
# # token length around 30 is enough (covered 99.15)

# # remove non-alphabet chars
# regex = re.compile(r'[^\w\s]')
# title = list(map(lambda x: regex.sub("", x), df["title"].to_list()))

# with open('preprocessed_title.pkl', 'wb') as f:
#     pickle.dump(title, f)

with open('./preprocessed_title.pkl', 'rb') as f:
    title = pickle.load(f)

title = random.sample(title, 2000)

# %% feed to the model
N_WORKERS = 16
BATCH_SIZE = 8
N_ITERS = 10
TOKEN_MAX_LENGTH = 30
SOM_ALPHA = 1
M = 50
N = 50

tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
pubMedDataset = utils.pubMedDataset(title,
                                    tokenizer,
                                    max_length = TOKEN_MAX_LENGTH, 
                                    batch_size = BATCH_SIZE)
pubMedLoader = torch.utils.data.DataLoader(
    pubMedDataset,
    batch_size = BATCH_SIZE,
    num_workers = N_WORKERS,
    shuffle = True,
)

som = utils.SapBERTembeddedSOM(
    m = M,
    n = N,
    dim = 768, # output dim from pretrained SapBERT
    niter = N_ITERS,
    batch_size = BATCH_SIZE
)


# train
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for iter_no in range(N_ITERS):
    loader = tqdm(enumerate(pubMedLoader), total = len(pubMedLoader))
    loader.set_description("Iter: {0}".format(iter_no))
    total_bmu_d = 0
    for i, data in loader:
        bmu_d = som(*data, iter_no)
        total_bmu_d += np.power(bmu_d, 2)
        loader.set_postfix(OrderedDict(bmu_d = bmu_d))
    
    som.save_model(iter_no)
    
    print("Average best unit distance^2: {0:.3f}".format(total_bmu_d / len(pubMedLoader)))
    grid = som.give_density_map(pubMedLoader)
    
# grid = som.give_density_map(pubMedLoader)
#%%