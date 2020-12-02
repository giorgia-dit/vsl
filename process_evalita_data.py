#!/usr/bin/env python3

import pickle
import argparse
import logging

from sklearn.model_selection import train_test_split


def my_args():

    args = argparse.Namespace()

    args.train = "./input/train_valid.txt"
    args.test = "./input/test.txt"
    args.ratio = 1
    return args

def get_args():
    parser = argparse.ArgumentParser(
        description='Data Preprocessing for Evalita')
    parser.add_argument('--train', type=str, default=None,
                        help='train data path')
    parser.add_argument('--test', type=str, default=None,
                        help='test data path')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='training data ratio')
    args = parser.parse_args()
    return args


def process_file(data_file):
    logging.info("loading data from " + data_file + " ...")
    sents = []
    tags = []
    with open(data_file, 'r', encoding='utf-8') as df:
        sent = []
        tag = []
        for line in df.readlines():
            if line.strip():
                index = line.find(' ')
                if index == -1:
                    raise ValueError('Format Error')
                sent.append(line[: index])
                tag.append(line[index + 1: -1])
            if line == "\n":
                sents.append(sent)
                tags.append(tag)
                sent = []
                tag = []
    return sents, tags


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')
    args = my_args()
    train_dev = process_file(args.train)
    test = list(process_file(args.test))
    train_sents, dev_sents, train_tags, dev_tags = \
            train_test_split(train_dev[0], train_dev[1], test_size=0.1)
    train = [train_sents, train_tags]
    dev = [dev_sents, dev_tags]

    tag_set = set(sum([sum(d[1], []) for d in [train, dev, test]],
                  []))
    with open(f"./input/preprocessed/evalita_tagfile", "w+", encoding='utf-8') as fp:
        fp.write('\n'.join(sorted(list(tag_set))))

    if args.ratio != 1:
        train_x, test_x, train_y, test_y = \
            train_test_split(train[0], train[1], test_size=args.ratio)
        train = [test_x, test_y]
        assert len(train_x) == len(train_y)

    logging.info("#train: {}".format(len(train[0])))
    logging.info("#dev: {}".format(len(dev[0])))
    logging.info("#test: {}".format(len(test[0])))

    dataset = {"train": train,
               "unlabel": [],
               "dev": dev,
               "test": test}

    pickle.dump(
        dataset, open(f"./input/preprocessed/pproc.evalita", "wb+"), protocol=-1)