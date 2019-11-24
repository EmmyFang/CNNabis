import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from model import LeafNet
from torchvision import datasets, models, transforms
from time import time
import os
import argparse


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


def write_record(record, args):
    record = np.transpose(record)

    df = pd.DataFrame({
        'step': record[0], 'time': record[1], 'epoch': record[2], 'train_loss': record[3],
        'train_error': record[4], 'valid_error': record[5]
    })
    df.to_csv(os.getcwd() + '/result/{}.csv'.format(get_name(args)), index=False)

    return


def evaluate(model, valid_loader):
    model.eval()
    correct = 0
    tot_size = 0
    for i, data in enumerate(valid_loader, 1):
        inputs, label = data[0].cuda(), data[1].cuda()
        outputs = model(inputs)
        _, predict = torch.max(outputs, 1)
        correct += torch.sum(predict == label)
        tot_size += len(label)

    return correct.double() / tot_size


def load_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    learning_rate = args.learning_rate
    freeze = args.freeze
    transfer = args.transfer
    model_name = args.model
    size = args.size

    hyper = {'width': size, 'ker_wid': 5, 'stride': 1, 'dilation': 1, 'padding': 0, 'pool': 2}
    output_size = 8

    if model_name == 'resnet':
        model = models.resnet18(pretrained=transfer)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, output_size)
            optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate, eps=1e-4)
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, output_size)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
    elif model_name == 'my':
        model = LeafNet(hyper)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    loss_fnc = nn.CrossEntropyLoss()

    model.to(device)

    return model, optimizer, scheduler, loss_fnc


def train(args, train_loader, valid_loader):
    epochs = args.epochs
    curr_path = os.getcwd()

    if not os.path.exists(curr_path + '/model'):
        os.makedirs(curr_path + '/model')

    if not os.path.exists(curr_path + '/result'):
        os.makedirs(curr_path + '/result')

    model, optimizer, scheduler, loss_fnc = load_model(args)

    steps = 0
    record = []
    max_acc = 0
    start = time()

    for epoch in range(epochs):
        model.train()
        scheduler.step()

        running_loss = 0.0
        running_correct = 0
        tot_size = 0
        for i, data in enumerate(train_loader, 1):

            steps += 1
            print(steps)
            inputs, label = data[0].cuda(), data[1].cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            _, predict = torch.max(outputs, 1)
            loss = loss_fnc(outputs, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(predict == label.data)
            tot_size += len(label)
            print('running corr: {}, tot size: {}'.format(running_correct, tot_size))
            print('train loader size: {}'.format(len(train_loader.dataset)))
        eval_acc = evaluate(model, valid_loader)

        if eval_acc >= max_acc:
            max_acc = eval_acc
            torch.save(model, curr_path + '/model/{}.pt'.format(get_name(args)))

        rec = [steps, time() - start, epoch + 1, running_loss, running_correct.double() / tot_size, eval_acc]
        record.append(rec)

        print('ep:', rec[2], ', los:', rec[3], ', acc:', rec[4], ', val:', rec[5])

    write_record(record, args)


def test(args, test_loader):
    model = torch.load(os.getcwd() + '/model/{}.pt'.format(get_name(args)))
    print('test accuracy is {}%'.format(evaluate(model, test_loader)))


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

    train_loader, valid_loader, test_loader = get_loader(args)

    train(args, train_loader, valid_loader)

    test(args, test_loader)
