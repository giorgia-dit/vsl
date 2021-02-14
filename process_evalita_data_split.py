#!/usr/bin/env python3

import pickle
import argparse
import logging

from sklearn.model_selection import train_test_split


def my_args():

    args = argparse.Namespace()

    args.train = "./input/train_valid.txt"
    args.test = "./input/test.txt"
    args.labratio = 0.2
    args.unlabratio = None
    return args

def get_args():
    parser = argparse.ArgumentParser(
        description='Data Preprocessing for Evalita')
    parser.add_argument('--train', type=str, default=None,
                        help='train data path')
    parser.add_argument('--test', type=str, default=None,
                        help='test data path')
    parser.add_argument('--labratio', type=float, default=1.0,
                        help='training labeled data ratio')
    parser.add_argument('--unlabratio', type=float, default=1.0,
                        help='training unlabeled data ratio')
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
    unlabel = []

    tag_set = set(sum([sum(d[1], []) for d in [train, dev, test]],
                  []))
    with open(f"./input/preprocessed/evalita_tagfile", "w+", encoding='utf-8') as fp:
        fp.write('\n'.join(sorted(list(tag_set))))

    if args.labratio != 1.0:
        train_sents, unlabel_sents, train_tags, unlabel_tags = \
            train_test_split(train[0], train[1],
                             train_size=args.labratio, test_size=args.unlabratio, shuffle=True)
        assert len(train_sents) == len(train_tags)
        assert len(unlabel_sents) == len(unlabel_tags)

        train = [train_sents, train_tags]
        unlabel = [unlabel_sents, unlabel_tags]


    logging.info("#train: {}".format(len(train[0])))
    logging.info("#unlabeled: {}".format(len(unlabel[0])))
    logging.info("#dev: {}".format(len(dev[0])))
    logging.info("#test: {}".format(len(test[0])))

    dataset = {"train": train,
               "unlabel": unlabel,
               "dev": dev,
               "test": test}

    output = "./input/preprocessed/pproc"
    if args.labratio != 1.0:
        output += f"_l{str(args.labratio)[-1]}"
    if args.unlabratio:
        output += f"_ul{str(args.unlabratio)[-1]}"
    output += ".evalita"

    pickle.dump(
        dataset, open(output, "wb+"), protocol=-1)