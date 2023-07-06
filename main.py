#%% import packages
import utils
import som
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

# title = random.sample(title, 500)
# title = title[:500]

# %% feed to the model
N_WORKERS = 16
BATCH_SIZE = 16
N_EPOCHS = 500
TOKEN_MAX_LENGTH = 30
SOM_LR = 0.01
M = 5
N = 5
PLOT_STEP = 100

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

somEmbedder = utils.SapBERTembedded()
# train
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embeddings = []
loader = tqdm(enumerate(pubMedLoader), total = len(pubMedLoader))
for i, data in loader:
    embedded = somEmbedder(*data)
    embeddings.extend(embedded)

embeddings = np.array(embeddings)
del somEmbedder

embedLoader = torch.utils.data.DataLoader(
    embeddings,
    batch_size = BATCH_SIZE,
    num_workers = N_WORKERS,
    shuffle = True
)

#%%
from importlib import reload 
som = reload(som)
# train
os.environ["TOKENIZERS_PARALLELISM"] = "false"
somNet = som.SOM(
    m = M,
    n = N,
    dim = 768,
    niter = len(embedLoader),
    epochs = N_EPOCHS,
    lr = SOM_LR,
)

for epoch in tqdm(range(N_EPOCHS), desc = "Training:"):
    carryover = epoch * len(embedLoader)
    for iter, batch_data in enumerate(embedLoader):
        somNet(batch_data, epoch)
    somNet.update()
    
    if epoch % PLOT_STEP == 0:
        somNet.give_density_map(embedLoader)
somNet.give_density_map(embedLoader)

#%%

