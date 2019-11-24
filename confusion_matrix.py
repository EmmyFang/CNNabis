import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
from model import LeafNet
from torchvision import datasets, models, transforms
from time import time
import os
import argparse
import math


def get_name(args):
    name = 'bs {} lr {} epochs {} model {} freeze {} transfer {} valid ratio {} shuffle {} seed {} size {}'.format(
        args.batch_size, args.learning_rate, args.epochs,
        args.model, args.freeze, args.transfer,
        args.valid_ratio, args.shuffle, args.seed, args.size
    )

    return name


def get_loader(args):
    valid_ratio = args.valid_ratio
    batch_size = args.batch_size
    shuffle = args.shuffle
    seed = args.seed
    size = args.size

    train_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomAffine(360),
        transforms.ToTensor(),
        transforms.Normalize([0.57281149, 0.58008847, 0.45508676], [0.33392794, 0.32032072, 0.37041936])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.57281149, 0.58008847, 0.45508676], [0.33392794, 0.32032072, 0.37041936])
    ])

    data_dir = os.getcwd() + '/data'
    train_datasets = datasets.ImageFolder(data_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(data_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(data_dir, transform=valid_transforms)

    length = len(train_datasets)
    indices = list(range(length))
    split = int(np.floor(valid_ratio * length))
    half = split // 2

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_ind, valid_ind, test_ind = indices[split:], indices[half:split], indices[:half]
    train_sam = SubsetRandomSampler(train_ind)
    valid_sam = SubsetRandomSampler(valid_ind)
    test_sam = SubsetRandomSampler(test_ind)

    train_loader = DataLoader(train_datasets, batch_size=batch_size, sampler=train_sam)
    valid_loader = DataLoader(valid_datasets, batch_size=batch_size, sampler=valid_sam)
    test_loader = DataLoader(test_datasets, batch_size=batch_size, sampler=test_sam)

    return train_loader, valid_loader, test_loader


def evaluate(model, valid_loader):
    model.eval()
    correct = 0
    tot_size = 0
    matrix = np.zeros((8, 8), dtype=int)
    probability = np.zeros((8, 8))
    images = np.zeros((8, 8, 3, 200, 200))
    for i, data in enumerate(valid_loader, 1):
        inputs, label = data[0], data[1]
        outputs = model(inputs)
        _, predict = torch.max(outputs, 1)
        predict = predict.squeeze()

        outputs = torch.nn.functional.softmax(outputs, dim=1).detach().numpy()

        for j, p in enumerate(predict):
            true_label = label[j]
            matrix[true_label][p] += 1

            if probability[true_label][p] < outputs[j][p]:
                images[true_label][p] = inputs[j].numpy()
                probability[true_label][p] = outputs[j][p]

        correct += torch.sum(predict == label)
        tot_size += len(label)

    return (correct.double() / tot_size), matrix, probability, images


def test(args, test_loader):
    model = torch.load(os.getcwd() + '/model/best.pt', map_location='cpu')
    test_accuracy, confusion_matrix, probability, images = evaluate(model, test_loader)
    print('test accuracy is {}%'.format(test_accuracy))
    return confusion_matrix, probability, images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--valid_ratio', type=float, default=0.2, help='validation set proportion')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--freeze', type=int, default=0, help='freeze parameters')
    parser.add_argument('--transfer', type=int, default=1, help='transfer learning')
    parser.add_argument('--model', type=str, default='resnet', help='resnet or my')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--shuffle', type=int, default=1, help='shuffle or not')
    parser.add_argument('--epochs', type=int, default=25, help='num of epochs')
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--size', type=int, default=200, help='image size')

    args = parser.parse_args()

    _, _, test_loader = get_loader(args)
    curr_path = os.getcwd()
    size = 200
    conf_matrix, probability, images = test(args, test_loader)
    print(conf_matrix)
    print(probability)
    # print(images)

    np.save('conf_matrix', conf_matrix)
    np.save('images', images)
    np.save('probability', probability)

