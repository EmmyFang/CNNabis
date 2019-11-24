from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os


def get_param(curr_path, size):
    data_transforms = transforms.Compose([
        transforms.Resize(size)
    ])
    data_dir = curr_path + '/data'
    dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    means = np.array([0, 0, 0])
    stdev = np.array([0, 0, 0])
    for i in dataset:
        image = i[0]
        means = means + np.squeeze(np.mean(image, (0, 1)))
    means = means / len(dataset)
    for i in dataset:
        image = i[0]
        image = image - means
        image = image * image
        stdev = stdev + np.squeeze(np.mean(image, (0, 1)))
    stdev = stdev / len(dataset)
    stdev = stdev ** 0.5
    return means / 255, stdev / 255


def get_image(curr_path, size, means=[0.57281149, 0.58008847, 0.45508676], stdev=[0.33392794, 0.32032072, 0.37041936]):
    user_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomAffine(360),
        transforms.ToTensor(),
        transforms.Normalize(means, stdev)
    ])
    user_dir = curr_path + '/data'
    dataset = datasets.ImageFolder(user_dir, transform=user_transforms)
    return dataset


def display(dataset):
    plt.figure()
    plt.imshow(np.transpose(dataset[0][0]))
    plt.savefig(os.getcwd() + "/augmented.png")


if __name__ == '__main__':
    curr_path = os.getcwd()
    # means, stdev = get_param(curr_path, 200)
    dataset = get_image(curr_path, 200)
    display(dataset)