import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os


def get_name(args):
    name = 'bs {} lr {} epochs {} model {} freeze {} transfer {} valid ratio {} shuffle {} seed {} size {}'.format(
        args.batch_size, args.learning_rate, args.epochs,
        args.model, args.freeze, args.transfer,
        args.valid_ratio, args.shuffle, args.seed, args.size
    )

    return name


def load_record(args):
    rec = pd.read_csv(os.getcwd() + "/result/{}.csv".format(get_name(args)))
    record = np.transpose(rec.values.tolist())
    return record


def plot_record(record, args):
    plt.figure()
    plt.title("{}".format(args))
    plt.plot(record[0], record[3], label='train')
    plt.plot(record[0], record[5], label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.savefig(os.getcwd() + "/result/{}.png".format(get_name(args)))


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

    record = load_record(args)

    plot_record(record, args)