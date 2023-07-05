import test
import utils
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_dataset():
    all_names = ["covid-19", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"] 
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
    pubMedDataset = utils.pubMedDataset(all_names, tokenizer, 30, 128)
    
    assert len(pubMedDataset) == 4
    assert len(pubMedDataset[1]) == 3


def test_getEmbed():
    batch_size = 2
    n_iters = 10
    n_workers = 4

    all_names = ["covid-19", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"] 
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext") 
    pubMedDataset = utils.pubMedDataset(all_names,
                                        tokenizer,
                                        max_length = 30, 
                                        batch_size = batch_size)
    pubMedLoader = torch.utils.data.DataLoader(
        pubMedDataset,
        batch_size = batch_size,
        num_workers = n_workers,
        shuffle = True,
    )
    som = utils.SapBERTembeddedSOM(
        m = 10,
        n = 10,
        dim = 768, # output dim from pretrained SapBERT
        niter = n_iters,
        batch_size = batch_size
    )

    # get inputs
    i, t, a = next(iter(pubMedLoader))
    embedded = som.get_embedded(i, t, a)

    assert embedded.size() == (2, 768)

def test_som():
    batch_size = 2
    n_iters = 5
    n_workers = 4

    all_names = ["covid-19", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"] 
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext") 
    pubMedDataset = utils.pubMedDataset(all_names,
                                        tokenizer,
                                        max_length = 30, 
                                        batch_size = batch_size)
    pubMedLoader = torch.utils.data.DataLoader(
        pubMedDataset,
        batch_size = batch_size,
        num_workers = n_workers,
        shuffle = True,
    )

    som = utils.SapBERTembeddedSOM(
        m = 5,
        n = 5,
        dim = 768, # output dim from pretrained SapBERT
        niter = n_iters,
        batch_size = batch_size
    )

    before = som.contents.cpu().detach().numpy()

    # train
    for iter_no in range(n_iters):
        for i, data in tqdm(enumerate(pubMedLoader)):
            _ = som(*data, iter_no)

    after = som.contents.cpu().detach().numpy()

    assert (before == after).all() == False


def test_map_vects():
    batch_size = 2
    n_iters = 5
    n_workers = 4

    all_names = ["covid-19", "Coronavirus infection", "high fever", "Tumor of posterior wall of oropharynx"] 
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext") 
    pubMedDataset = utils.pubMedDataset(all_names,
                                        tokenizer,
                                        max_length = 30, 
                                        batch_size = batch_size)
    pubMedLoader = torch.utils.data.DataLoader(
        pubMedDataset,
        batch_size = batch_size,
        num_workers = n_workers,
        shuffle = True,
    )

    som = utils.SapBERTembeddedSOM(
        m = 10,
        n = 10,
        dim = 768, # output dim from pretrained SapBERT
        niter = n_iters,
        batch_size = batch_size
    )

    mapped = []
    for i, data in tqdm(enumerate(pubMedLoader)):
        out = som.map_vects(*data)
        print(out)
        mapped.extend(out)
    
    assert np.array(mapped).shape == (4,)