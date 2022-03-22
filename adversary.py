#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import os
import torch
from tqdm import tqdm
from torch.distributions.uniform import Uniform
from evalution import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = ""


# FGSM training class
class FGSM(object):

    def __init__(self, model, embed_name, epsilon=8/255, alpha=10/255):
        '''
        model: model class
        embed_name: name of embedding
        epsilon: radius
        alpha: step size
        '''
        self.model = model
        self.embed_name = embed_name
        self.epsilon = torch.tensor([epsilon])
        self.__alpha = alpha
        # initialize delta from uniform distribution
        self.__uniform_init = Uniform(-self.epsilon, self.epsilon)

        self.backup = {}

    def attack(self):
        '''
        delta = Uniform(-epsilon, epsilon)
        delta = delta + alpha * sign(gradient of delta)
        delta = max(min(delta, epsilon), -epsilon)
        '''
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.embed_name in name:
                self.backup[name] = param.data.clone()
                delta = self.__uniform_init.sample()
                delta.requires_grad = True
                delta = delta + self.__alpha * torch.sign(param.grad)
                delta = torch.max(torch.min(delta, self.epsilon), -self.epsilon)
                param.data.add_(delta)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.embed_name in name:
                assert name in self.backup
                param.data = self.backup[name]

        self.backup = {}


# PGD training class
class PGD(object):

    def __init__(self, model, embed_name, epsilon=8/255, alpha=10/255):
        self.__model = model
        self.__embed_name = embed_name
        self.__epsilon = torch.tensor([epsilon])
        self.__alpha = alpha

        self.__embed_backup = {}
        self.__grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.__model.named_parameters():
            if param.requires_grad and self.__embed_name in name:
                if is_first_attack:
                    self.__embed_backup[name] = param.data.clone()

                param.data = self.project(name, param)
                

    def restore(self):
        for name, param in self.__model.named_parameters():
            if param.requires_grad and self.__embed_name in name:
                assert name in self.__embed_backup
                param.data = self.__embed_backup[name]
        
        self.__embed_backup = {}

    def project(self, param_name, param):
        '''
        delta = delta of last PGD step
        delta = delta + alpha * sign(gradient of delta)
        delta = max(min(delta, epsilon), -epsilon)
        '''
        delta = param.data - self.__embed_backup[param_name]
        delta = delta + self.__alpha * torch.sign(param.grad)
        delta = torch.max(torch.min(delta, self.__epsilon), -self.__epsilon)

        return self.__embed_backup[param_name] + delta

    def backup_grad(self):
        for name, param in self.__model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.__grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.__model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.__grad_backup[name]


# normal training function
def normal_train(
    model, train_iter, test_iter, save_path, 
    optimizer, epochs=20
    ):

    loss_fn = torch.nn.CrossEntropyLoss()

    for i in tqdm(range(1, epochs+1)):
        print(f'Epoch {i}:')
        model.train()

        for X, y in tqdm(train_iter):

            loss = model(X)
            loss = loss_fn(loss, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
        # validation
        valid_loss, valid_acc = evaluate(test_iter, model, loss_fn)
        print(f'loss = {valid_loss}, acc = {valid_acc}')
        
        # save model checkpoint file every 5 epochs
        if i % 5 == 0:
            torch.save(model.state_dict(), f'{save_path}/normal_model-{i}.pth')
            print(f'Model checkpoint {i} saved!')


# FGSM training function
def fgsm_train(
    model, train_iter, test_iter, save_path, 
    optimizer, epochs=20, embed_name='embeddings.'
    ):

    loss_fn = torch.nn.CrossEntropyLoss()
    fgsm = FGSM(model, embed_name)
    
    for i in tqdm(range(1, epochs+1)):
        print(f'Epoch {i}:')
        model.train()

        for X, y in tqdm(train_iter):
            # normal training steps
            loss = model(X)
            loss = loss_fn(loss, y)
            loss.backward()

            # FGSM adversarial training steps
            fgsm.attack()
            adv_loss = model(X)
            adv_loss = loss_fn(adv_loss, y)
            adv_loss.backward()
            fgsm.restore()

            # update weights
            optimizer.step()
            optimizer.zero_grad()
        
        # validation
        valid_loss, valid_acc = evaluate(test_iter, model, loss_fn)
        print(f'loss = {valid_loss}, acc = {valid_acc}')

        # save model checkpoint file every 5 epochs
        if i % 5 == 0:
            torch.save(model.state_dict(), f'{save_path}/fgsm_model-{i}.pth')
            print(f'Model checkpoint {i} saved!')


# PGD training function
def pgd_train(
    model, train_iter, test_iter, save_path,
    optimizer, replays=3, embed_name='embeddings.', epochs=20):
    
    loss_fn = torch.nn.CrossEntropyLoss()
    pgd = PGD(model, embed_name)
    
    for i in tqdm(range(1, epochs+1)):
        print(f'Epoch {i}:')
        model.train()

        for X, y in tqdm(train_iter):
            # normal training steps
            loss = model(X)
            loss = loss_fn(loss, y)
            loss.backward()
            pgd.backup_grad()

            # PGD adversarial training steps
            for t in range(replays):
                pgd.attack(is_first_attack=(t==0))
                if t != replays-1:
                    optimizer.zero_grad()
                else:
                    pgd.restore_grad()
            
                adv_loss = model(X)
                adv_loss = loss_fn(adv_loss, y)
                adv_loss.backward()
        
            pgd.restore()

            # update weights
            optimizer.step()
            optimizer.zero_grad()

        # validation
        valid_loss, valid_acc = evaluate(test_iter, model, loss_fn)
        print(f'loss = {valid_loss}, acc = {valid_acc}')

        # save model checkpoint file every 5 epochs
        if i % 5 == 0:
            torch.save(model.state_dict(), f'{save_path}/pgd_model-{i}.pth')
            print(f'Model checkpoint {i} saved!')


# free training function
def free_train(
    model, train_iter, test_iter, 
    optimizer,loss_fn, save_path, 
    replays=5, epochs=5
    ):
    
    epsilon = torch.tensor([8/255])
    
    model.perturbation = torch.tensor([0.0])
    model.perturbation.requires_grad = True
    
    for i in tqdm(range(1, epochs+1)):
        print(f'Epoch {i}:')
        model.train()
        
        for X, y in tqdm(train_iter):
            for _ in range(replays):
                loss = model(X)
                loss = loss_fn(loss, y)
                
                optimizer.zero_grad()
                loss.backward()
                
                grad = model.embeddings.weight.grad[:X.size(1)]
                model.perturbation = model.perturbation + epsilon * torch.sign(grad)
                model.perturbation.data = torch.max(
                    torch.min(model.perturbation, epsilon), -epsilon
                    )
                
                optimizer.step()
                grad.zero_()
            
        # validation
        valid_loss, valid_acc = evaluate(test_iter, model, loss_fn)
        print(f'loss = {valid_loss}, acc = {valid_acc}')
        
        # save model checkpoint file
        torch.save(model.state_dict(), f'{save_path}/free_model-{i}.pth')
        print(f'Model checkpoint {i} saved!')
