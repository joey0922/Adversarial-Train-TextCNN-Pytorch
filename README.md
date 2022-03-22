# Adversarial-Train-TextCNN-Pytorch
A repository that implements three adversarial training methods, FGSM, "Free" and PGD, which are published in the paper, [*"Fast is better than free: Revisiting adversarial training"*](https://arxiv.org/abs/2001.03994), created by [*Eric Wong*](https://riceric22.github.io/),  [*Leslie Rice*](https://leslierice1.github.io/), and [*Zico Kolter*](http://zicokolter.com/). Then leverage them to train textCNN models, and show how they performance comparing with an original textCNN model.
# Implementation

## 1. FGSM

## 2. Free

## 3. PGD

# Configuration
## Environment
+ joblib==1.0.1
+ numpy==1.20.1
+ pandas==1.2.4
+ scikit_learn==1.0.2
+ torch==1.11.0
+ tqdm==4.59.0
## Data
|     | 数量 |
|-----|-----|
|训练集|180000|
|验证集|10000|
|测试集|10000|
# Result							

# Reference
 * [Fast is better than free: Revisiting adversarial training](https://arxiv.org/abs/2001.03994)
 * [fast_adversarial](https://github.com/locuslab/fast_adversarial)
 * [Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)
 * [知乎专栏：【炼丹技巧】功守道：NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)
 * [知乎专栏：一文搞懂NLP中的对抗训练](https://zhuanlan.zhihu.com/p/103593948)
