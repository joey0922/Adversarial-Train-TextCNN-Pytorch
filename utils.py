#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import re
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from config import TextCNNConfig
from textCNN import TextCNN, AdverTextCNN


# build model from scratch
def build_model(embeddings=None, config=None, n_class=None, adver=False):

    cnn_config = TextCNNConfig() if config is None else config

    # guarantee that embeddings.shape[0] is equal to vocabulary size
    if embeddings is not None:
        vocab_size, embed_dim = embeddings.shape
        cnn_config.from_pretrained = embeddings
        # initialize real dimension of embedding
        cnn_config.embed_dim = embed_dim
        # initialize real volume of vocabulary
        cnn_config.vocab_size = vocab_size

    if n_class:
        cnn_config.n_classes = n_class

    return TextCNN(cnn_config) if not adver else AdverTextCNN(cnn_config)


# build model and initilize it from checkpoint
def load_model(model_path, adver=False):

    model = build_model(adver=adver)
    model.load_state_dict(torch.load(model_path))

    return model


# get texts and their labels from txt file
def load_dataset(fpath, vocab_dict, pad_size=32):
    
    line_ids_list = []
    label_list = []

    with open(fpath, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            line, label = re.split('\t', line)  
            line_ids = [vocab_dict.get(char, vocab_dict.get('<UNK>')) for char in line]
            
            if pad_size:
                if len(line_ids) >= pad_size:
                    line_ids = line_ids[:pad_size]
                else:
                    line_ids.extend([vocab_dict.get('<PAD>', 0)] * (pad_size - len(line_ids)))
                    
            line_ids_list.append(line_ids)
            label_list.append(int(label))
            
    return line_ids_list, label_list
    

# build data loader
def build_data_loader(fpath, vocab_dict, batch_size=128, pad_size=32):
    
    inps, labels = load_dataset(fpath, vocab_dict, pad_size)
    inps, labels = torch.tensor(inps), torch.tensor(labels)
    dataset = TensorDataset(inps, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return data_loader


# get vocabulary embeddings from sogou new char
def build_embeddings(embed_path, vocab_dict):
    
    with open(embed_path, 'r', encoding='utf-8') as fin:
        dim = int(re.split(' ', fin.readline())[-1])
        embeddings = np.random.rand(len(vocab_dict), dim)
        
        num = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue
                
            line = re.split(' ', line)
            if line[0] in vocab_dict.keys():
                num += 1
                embeddings[vocab_dict[line[0]]] = np.asarray(line[1:], dtype='float32')
                
    # if len(vocab_dict) - num - 2 > 0:
    #     print(f'There are {len(vocab_dict) - num - 2} words not included!')
    
    return embeddings