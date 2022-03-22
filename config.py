#!/usr/bin/env python
# -*- encoding:utf-8 -*-


class TextCNNConfig:

    def __init__(self):

        self.from_pretrained = None  # pretrained embeddings
        self.embed_dim = 300  # embedding dimension, should initialize before training
        self.vocab_size = 4762  # vocabulary volume, should initialize before training
        self.n_classes = 10  # number of class, should initialize before training

        self.kernel_sizes = (2, 3, 4)  # kernerl size list
        self.n_filters = 256  # number of filter

        self.dropout=0.5  # dropout probability


class TrainConfig:

    def __init__(self):

        self.epochs = 20
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.optimizer = None
        self.loss_function = None
        self.pad_size = 32  # maximal length of a sequence

        self.replays = 3  # how many times pgd and free trained per mini-batch

        self.vocab_path = './data/vocab.pkl'
        self.embed_path = './data/embeddings.npy' # embeddings of vocabulary
        self.train_path = './data/train.txt'
        self.valid_path = './data/dev.txt'

        self.ckpt_path = None  # path to model checkpoint saved