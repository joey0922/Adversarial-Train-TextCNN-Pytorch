#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F


# normal textcnn class
class TextCNN(nn.Module):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        
        # embedding layer
        if config.from_pretrained is not None:
            self.embeddings = nn.Embedding.from_pretrained(
                config.from_pretrained, 
                freeze=False
                )
        else:
            self.embeddings = nn.Embedding(
                config.vocab_size, 
                config.embed_dim, 
                padding_idx=config.vocab_size - 1
                )

        # convolution layer
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.n_filters, (k, config.embed_dim)) 
            for k in config.kernel_sizes]
            )
        
        # dropout
        self.dropout = nn.Dropout(config.dropout)
        # full connection layer
        self.dense = nn.Linear(
            config.n_filters * len(config.kernel_sizes), 
            config.n_classes
            )

    def conv_pool(self, x, conv):

        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, x):

        logits = self.embeddings(x)
        logits = logits.unsqueeze(1)
        logits = torch.cat([self.conv_pool(logits, conv) for conv in self.convs], 1)
        logits = self.dropout(logits)
        logits = self.dense(logits)

        return logits


# textcnn class with perturbation
class AdverTextCNN(TextCNN):
    
    def __init__(self, config):
        super(AdverTextCNN, self).__init__(config)
        self.perturbation = None
        
    def forward(self, x):
        logits = self.embeddings(x)
        if self.perturbation is not None:
            logits = logits + self.perturbation
        logits = logits.unsqueeze(1)
        logits = torch.cat([self.conv_pool(logits, conv) for conv in self.convs], 1)
        logits = self.dropout(logits)
        logits = self.dense(logits)

        return logits