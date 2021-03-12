#!/usr/bin/env python3

import pickle
import argparse
import logging

from sklearn.model_selection import train_test_split


def my_args():
    file = 'it_postwita-ud-'
    output_dir = 'input/preprocessed'

    args = argparse.Namespace()

    args.train = f"./input/{file}train.conllu"
    args.dev = f"./input/{file}dev.conllu"
    args.test = f"./input/{file}test.conllu"
    args.output = f"./{output_dir}/{file}pproc"
    args.labratio = 0.2
    args.unlabratio = 0.8
    return args


def get_args():
    parser = argparse.ArgumentParser(
        description='Data Preprocessing for Universal Dependencies')
    parser.add_argument('--train', type=str, default=None,
                        help='train data path')
    parser.add_argument('--dev', type=str, default=None,
                        help='dev data path')
    parser.add_argument('--test', type=str, default=None,
                        help='test data path')
    parser.add_argument('--output', type=str, default=None,
                        help='output file name')
    parser.add_argument('--labratio', type=float, default=1.0,
                        help='labeled data ratio')
    parser.add_argument('--unlabratio', type=float, default=0.0,
                        help='unlabeled data ratio')
    args = parser.parse_args()
    return args


def load_data(data_file):
    logging.info("loading data from {} ...".format(data_file))
    sents = []
    tags = []
    sent = []
    tag = []
    with open(data_file, 'r', encoding="utf-8") as f:
        for line in f:
            if line[0] == "#":
                continue
            if line.strip():
                token = line.strip("\n").split("\t")
                word = token[1]
                t = token[3]
                sent.append(word)
                tag.append(t)
            else:
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
    logging.info("##### training data #####")
    all_sents, all_tags = load_data(args.train)

    filtered_sents = []
    filtered_tags = []

    for sent, tag in zip(all_sents, all_tags):
        if len(sent) <= 80:
            filtered_sents.append(sent)
            filtered_tags.append(tag)
    all_sents = filtered_sents
    all_tags = filtered_tags

    logging.info("random splitting training data with ratio of {}..."
                 .format(args.labratio))

    if args.labratio == 1.0:
        train_sents, unlabel_sents, train_tags, unlabel_tags = all_sents, [], all_tags, []
    else:
        train_sents, unlabel_sents, train_tags, unlabel_tags = \
            train_test_split(all_sents, all_tags,
                             train_size=args.labratio, test_size=args.unlabratio, shuffle=True)

    # todo check: statistics
    mean_length = sum([len(s) for s in all_sents])/len(all_sents)
    max_length = max([len(s) for s in all_sents])
    logging.info(f"mean length of sentences in dataset: {mean_length}\n"
                 f"max length of sentences in dataset: {max_length}")



    logging.info("#train sents: {}, #train words: {}, #train tags: {}"
                 .format(len(train_sents), len(sum(train_sents, [])),
                         len(sum(train_tags, []))))
    logging.info("#unlabeled sents: {}"
                 .format(len(unlabel_sents)))
    logging.info("##### dev data #####")
    dev_sents, dev_tags = load_data(args.dev)
    logging.info("#dev sents: {}, #dev words: {}, #dev tags: {}"
                 .format(len(dev_sents), len(sum(dev_sents, [])),
                         len(sum(dev_tags, []))))
    logging.info("##### test data #####")
    test_sents, test_tags = load_data(args.test)
    logging.info("#dev sents: {}, #dev words: {}, #dev tags: {}"
                 .format(len(test_sents), len(sum(test_sents, [])),
                         len(sum(test_tags, []))))
    output = "data" if args.output is None else args.output
    if args.labratio != 1.0:
        output += f"_l{str(args.labratio)[-1]}"
    if args.unlabratio:
        output += f"_ul{str(args.unlabratio)[-1]}"
    output += ".ud"

    tag_set = set(sum([sum(d, []) for d in [all_tags, dev_tags, test_tags]],
                      []))
    output_dir = '.' if args.output is None else output.rsplit('/', maxsplit=1)[0]
    with open(f"{output_dir}/ud_tagfile", "w+", encoding='utf-8') as fp:
        fp.write('\n'.join(sorted(list(tag_set))))
    dataset = {"train": [train_sents, train_tags],
               "unlabel": [unlabel_sents, unlabel_tags],
               "dev": [dev_sents, dev_tags],
               "test": [test_sents, test_tags]}
    pickle.dump(dataset, open(output, "wb+"), protocol=-1)
    logging.info("data saved to {}".format(output))

    ## statistics computation for test

    # logging.info("statistics on test tags below: \n")
    #
    # k_counts = {}
    # for t in tag_set:
    #     if t != "_":
    #         k_counts[t] = 0
    #         for i in test_tags:
    #             for j in i:
    #                 if j == t:
    #                     k_counts[t] += 1
    #
    # temp_log = ''
    # temp_tot = sum(k_counts.values())
    # temp_prop = {}
    # for k, c in sorted(k_counts.items()):
    #     temp_prop[k] = (c / temp_tot) * 100
    #     temp_log += f"({k},{c},{temp_prop[k]:.2f}) \n"
    # max_key = max(temp_prop, key=temp_prop.get)
    # temp_log += f"The tag occurring the most is: {max_key}, with rate {temp_prop[max_key]:.3f}"
    # logging.info(temp_log)


    #statistic computation for training

    logging.info("statistics on train tags below: \n")

    k_counts = {}
    for t in tag_set:
        if t != "_" and t != "X":
            k_counts[t] = 0
            for i in train_tags:
                for j in i:
                    if j == t:
                        k_counts[t] += 1

    temp_log = ''
    temp_tot = sum(k_counts.values())
    temp_prop = {}
    for k, c in sorted(k_counts.items()):
        temp_prop[k] = (c / temp_tot) * 100
        temp_log += f"({k},{c},{temp_prop[k]:.2f}) \n"
    max_key = max(temp_prop, key=temp_prop.get)
    temp_log += f"The tag occurring the most is: {max_key}, with rate {temp_prop[max_key]:.3f}"
    logging.info(temp_log)

