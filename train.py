#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import os
import joblib
import torch
import numpy as np
from adversary import normal_train, fgsm_train, pgd_train, free_train
from config import TrainConfig
from utils import build_data_loader, build_model

os.environ['CUDA_VISIBLE_DEVICES'] = ""


if __name__ == '__main__':
    
    train_config = TrainConfig()
    vocab_dict = joblib.load(train_config.vocab_path)
    # if there are no embeddings of vocabulary, 
    # use function build_embeddsings in utils.py to construct it
    embeddings = torch.FloatTensor(np.load(train_config.embed_path))

    # load data
    train_loader = build_data_loader(
        train_config.train_path, vocab_dict, 
        batch_size=train_config.batch_size, 
        pad_size=train_config.pad_size
    )
    dev_loader = build_data_loader(
        train_config.valid_path, vocab_dict, 
        batch_size=train_config.batch_size, 
        pad_size=train_config.pad_size
    )

    # build model
    # model = build_model(embeddings=embeddings)
    # optimizer = torch.optim.Adam(
    #     model.parameters(), 
    #     lr=train_config.learning_rate
    #     )

    # train_config.ckpt_path = './models/temp'
    # normal_train(model, train_loader, dev_loader, train_config.ckpt_path, optimizer)
    # fgsm_train(model, train_loader, dev_loader, train_config.ckpt_path, optimizer)
    # pgd_train(model, train_loader, dev_loader, train_config.ckpt_path, optimizer)

    # build free model
    model = build_model(embeddings=embeddings, adver=True)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=train_config.learning_rate
        )
    train_config.ckpt_path = './models/temp'
    loss_fn = torch.nn.CrossEntropyLoss()
    free_train(model, train_loader, dev_loader, optimizer, loss_fn, train_config.ckpt_path)
