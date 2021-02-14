#!/usr/bin/env python3

import pickle
import argparse
import logging
import csv
import emoji
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)

def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI

from sklearn.model_selection import train_test_split

# {1.0 isdt = 0.029 coris; 1.0 evalita = 0.013 coris; 1.0 postwita = 0.017}

def my_args():
    args = argparse.Namespace()
    args.set = 'twita'  # {'coris', 'twita'}
    args.ratio = 0.034
    return args

def get_args():
    parser = argparse.ArgumentParser(
        description='Data Preprocessing for Evalita')
    parser.add_argument('--set', type=str, default=None,
                        help='unlabeled set')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='ratio of set needed')
    args = parser.parse_args()
    return args


def process_file(set):
    if set == 'coris':
        data_file = "./input/unlab_mon0507.txt"

        logging.info("loading data from " + data_file + " ...")

        sents = []

        with open(data_file, 'r', encoding='latin-1') as df:
            sent = []
            for line in df.readlines()[1:]:
                if line.strip() == "<s>":
                    continue
                if line.strip() != "</s>":
                    token = line.strip().split("\t")
                    word = token[0]
                    sent.append(word)
                else:
                    sents.append(sent)
                    sent = []
        return sents


    elif set == 'twita':
        data_file = "./input/unlab_twita20.csv"

        logging.info("loading data from " + data_file + " ...")

        sents = []

        with open(data_file, newline='') as csvfile:
            tweets = csv.reader(csvfile, delimiter=',')
            tweets = list(tweets)
            for tw in tweets[1:]:
                sent = tw[1]
                sent = tknzr.tokenize(sent)
                emos = [w for w in sent if char_is_emoji(w)]
                for e in emos:
                    sent.remove(e)
                sents.append(sent)
        return sents


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')
    args = my_args()

    unlabel_sents = process_file(args.set)

    train, unlabel = \
            train_test_split(unlabel_sents, test_size=args.ratio, shuffle=True)

    logging.info("#unlabel: {}".format(len(unlabel)))

    output = f"./input/preprocessed/unlabel.{args.set}"
    output += f"_{str(args.ratio)[-2:]}"

    pickle.dump(
        unlabel, open(output, "wb+"), protocol=-1)

    logging.info("data saved to {}".format(output))
