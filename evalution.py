#!/usr/bin/env python
# -*- encoding:utf-8 -*-


import os
import joblib
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from utils import build_data_loader, load_model

import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = ""


def evaluate(test_loader, model, loss_fn):

    total_loss = 0
    acc = 0
    n = 0

    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            output = model(X)
            loss = loss_fn(output, y)
            total_loss += loss.item() * y.size(0)
            acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    
    return total_loss/n, acc/n


def test(model, test_loader):

    model.eval()
    with torch.no_grad():
        y_pred, y_true = list(zip(*((model(X).argmax(dim=1), y) for X,y in test_loader)))
    
    y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)
    accuracy = ((y_pred == y_true).sum() / y_pred.shape[0]).item()

    matrix = confusion_matrix(y_true, y_pred)
    # sum of each predicted label
    pred_all = np.sum(matrix, axis=0)
    # sum of each real label
    true_all = np.sum(matrix, axis=1)
    # sum of ecah label predicted correctly
    pred_correct = np.diagonal(matrix)

    precision = pred_correct/pred_all
    recall = pred_correct/true_all
    f1 = precision * recall / precision + recall

    return accuracy*100, precision*100, recall*100, f1


def predict(model):
    pass


if __name__ == '__main__':
    
    # load vocabulary
    vocab_path = './data/vocab.pkl'
    vocab_dict = joblib.load(vocab_path)

    label_path = './data/class.txt'
    with open(label_path, 'r', encoding='utf-8') as fin:
        label_list = [line.strip() for line in fin]

    # load data
    test_path = './data/test.txt'
    test_loader = build_data_loader(
        test_path, vocab_dict, 
        batch_size=128, 
        pad_size=32
    )

    # # initialize model
    # model_path = './model/normal_model.pth'
    # model = load_model(model_path, adver=False)

    # accuracy, precision, recall, f1 = test(model, test_loader)

    # df = pd.DataFrame(index=label_list)
    # df['precision'] = precision
    # df['recall'] = recall
    # df['f1'] = f1
    # df.loc['mean'] = [precision.mean(), recall.mean(), f1.mean()]
    # print(df)

    model_list = (('normal', False), ('fgsm', False), ('free', True), ('pgd', False))
    for style, adv in model_list:
        model_path = f'./model/{style}_model.pth'
        model = load_model(model_path, adver=adv)
        accuracy, precision, recall, f1 = test(model, test_loader)

        df = pd.DataFrame(index=label_list)
        df['precision'] = precision
        df['recall'] = recall
        df['f1'] = f1
        df.loc['mean'] = [precision.mean(), recall.mean(), f1.mean()]

        save_path = f'./output/{style}.xlsx'
        df.to_excel(save_path)
        print(f'{style} accuracy = {accuracy}')