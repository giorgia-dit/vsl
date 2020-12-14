import os
import pickle

import numpy as np
import torch

from collections import Counter

from decorators import auto_init_args, lazy_execute
from config import UNK_WORD_IDX, UNK_WORD, UNK_CHAR_IDX, \
    UNK_CHAR

from bert_features import Bert


class data_holder:
    @auto_init_args
    def __init__(self, train, dev, test, unlabel,
                 tag_vocab, vocab, char_vocab):
        self.inv_vocab = vocab
        self.inv_tag_vocab = {i: w for w, i in tag_vocab.items()}


class data_processor:
    def __init__(self, experiment):
        self.expe = experiment

    def process(self):
        fn = "vocab_" + str(self.expe.config.vocab_size)
        vocab_file = os.path.join(self.expe.config.vocab_file, fn)

        self.expe.log.info("loading data from {} ...".format(
            self.expe.config.data_file))
        with open(self.expe.config.data_file, "rb+") as infile:
            if self.expe.config.embed_type == 'ud' or self.expe.config.embed_type == 'bert':
                dataset = pickle.load(infile)
                train_data = dataset['train']
                dev_data = dataset['dev']
                test_data = dataset['test']

                # TODO FIX
                # train_data[0] = train_data[0][:3]
                # train_data[1] = train_data[1][:3]
                # dev_data[0] = dev_data[0][:3]
                # dev_data[1] = dev_data[1][:3]
                # test_data[0] = test_data[0][:3]
                # test_data[1] = test_data[1][:3]

            else:
                train_data, dev_data, test_data = pickle.load(infile)
        train_v_data = train_data[0]
        unlabeled_data = None

        if self.expe.config.use_unlabel:
            if self.expe.config.embed_type == 'ud' or self.expe.config.embed_type == 'bert':
                unlabeled_data = dataset['unlabel'][0]

                # TODO FIX
                # unlabeled_data = unlabeled_data[:3]

            else:
                unlabeled_data = self._load_sent(self.expe.config.unlabel_file)

            train_v_data = train_data[0] + unlabeled_data

        n_label_sents = len(train_data[0])


        W, vocab, char_vocab, n_sents, n_words = \
            self._build_vocab_from_embedding(
                train_v_data, n_label_sents, dev_data[0], test_data[0],
                self.expe.config.embed_file,
                self.expe.config.vocab_size, self.expe.config.char_vocab_size,
                file_name=vocab_file)

        tag_vocab = self._load_tag(self.expe.config.tag_file)

        self.expe.log.info("tag vocab size: {}".format(len(tag_vocab)))

        train_data = self._label_to_idx(
            train_data[0], train_data[1], vocab, char_vocab, tag_vocab, n_words, 'train')
        dev_data = self._label_to_idx(
            dev_data[0], dev_data[1], vocab, char_vocab, tag_vocab, n_words, 'dev')
        test_data = self._label_to_idx(
            test_data[0], test_data[1], vocab, char_vocab, tag_vocab, n_words, 'test')

        def cal_stats(data):
            unk_count = 0
            total_count = 0
            leng = []
            for sent in data:
                leng.append(len(sent))
                for w in sent:
                    if w == UNK_WORD_IDX:
                        unk_count += 1
                    total_count += 1
            return (unk_count, total_count, unk_count / total_count), \
                (len(leng), max(leng), min(leng), sum(leng) / len(leng))

        train_unk_stats, train_len_stats = cal_stats(train_data[0])
        self.expe.log.info("#train data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}"
                           .format(*train_len_stats))

        self.expe.log.info("#unk in train sentences: {}"
                           .format(train_unk_stats))

        dev_unk_stats, dev_len_stats = cal_stats(dev_data[0])
        self.expe.log.info("#dev data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}"
                           .format(*dev_len_stats))

        self.expe.log.info("#unk in dev sentences: {}"
                           .format(dev_unk_stats))

        test_unk_stats, test_len_stats = cal_stats(test_data[0])
        self.expe.log.info("#test data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}"
                           .format(*test_len_stats))

        self.expe.log.info("#unk in test sentences: {}"
                           .format(test_unk_stats))

        if self.expe.config.use_unlabel:
            unlabeled_data = self._unlabel_to_idx(
                unlabeled_data, vocab, char_vocab, n_words)
            un_unk_stats, un_len_stats = cal_stats(unlabeled_data[0])
            self.expe.log.info("#unlabeled data: {}, max len: {}, "
                               "min len: {}, avg len: {:.2f}"
                               .format(*un_len_stats))

            self.expe.log.info("#unk in unlabeled sentences: {}"
                               .format(un_unk_stats))

        data = data_holder(
            train=train_data,
            dev=dev_data,
            test=test_data,
            unlabel=unlabeled_data,
            tag_vocab=tag_vocab,
            vocab=vocab,
            char_vocab=char_vocab)

        return data, W

    def _load_tag(self, path):
        self.expe.log.info("loading tags from " + path)
        tag = {}
        with open(path, 'r') as f:
            for (n, i) in enumerate(f):
                tag[i.strip()] = n
        return tag

    def _load_sent(self, path):
        self.expe.log.info("loading data from " + path)
        sents = []
        with open(path, "r+", encoding='utf-8') as df:
            for line in df:
                if line.strip():
                    words = line.strip("\n").split(" ")
                    sents.append(words)
        return sents

    def _label_to_idx(self, sentences, tags, vocab, char_vocab, tag_vocab, n_words, partition):
        sentence_holder = []
        sent_char_holder = []
        tag_holder = []

        if partition == 'train':
            partition_range = list(range(1, n_words['label'] + 1))
        elif partition == 'dev':
            if 'unlabel' in n_words:
                zero_ind = n_words['label'] + n_words['unlabel'] + 1
                partition_range = list(range(zero_ind, zero_ind + n_words['dev']))
            else:
                zero_ind = n_words['label'] + 1
                partition_range = list(range(zero_ind, zero_ind + n_words['dev']))
        else:
            if 'unlabel' in n_words:
                zero_ind = n_words['label'] + n_words['unlabel'] + n_words['dev'] + 1
                partition_range = list(range(zero_ind, zero_ind + n_words['test']))
            else:
                zero_ind = n_words['label'] + n_words['dev'] + 1
                partition_range = list(range(zero_ind, zero_ind + n_words['test']))

        n = 0
        for sentence, tag in zip(sentences, tags):
            chars = []
            words = []
            for w in sentence:
                words.append(partition_range[n])
                n += 1
                chars.append([char_vocab.get(c, 0) for c in w])
            sentence_holder.append(words)
            sent_char_holder.append(chars)
            tag_holder.append([tag_vocab[t] for t in tag])

        self.expe.log.info("#sent: {}".format(len(sentence_holder)))
        self.expe.log.info("#word: {}".format(len(sum(sentence_holder, []))))
        self.expe.log.info("#tag: {}".format(len(sum(tag_holder, []))))

        return np.asarray(sentence_holder), np.asarray(sent_char_holder), \
            np.asarray(tag_holder)

    def _unlabel_to_idx(self, sentences, vocab, char_vocab, n_words):
        zero_ind = n_words['label'] + 1
        partition_range = list(range(zero_ind, zero_ind + n_words['label'] + 1))

        sentence_holder = []
        sent_char_holder = []

        n = 0
        for sentence in sentences:
            words = []
            chars = []
            for w in sentence:
                words.append(partition_range[n])
                n += 1
                chars.append([char_vocab.get(c, 0) for c in w])
            sentence_holder.append(words)
            sent_char_holder.append(chars)
        self.expe.log.info("#sent: {}".format(len(sentence_holder)))
        return np.asarray(sentence_holder), np.asarray(sent_char_holder)

# Le due funzioni precedenti trasformano i dataset di frasi / tag in indici corrispondenti alle posizioni
# di ogni parola / carattere / tag nei corrispondenti vocabolari

    def _load_twitter_embedding(self, path):
        with open(path, 'r', encoding='utf8') as fp:
            # word_vectors: word --> vector
            word_vectors = {}
            for line in fp:
                line = line.strip("\n").split("\t")
                word_vectors[line[0]] = np.array(
                    list(map(float, line[1].split(" "))), dtype='float32')
        vocab_embed = word_vectors.keys()
        embed_dim = word_vectors[next(iter(vocab_embed))].shape[0]

        return word_vectors, vocab_embed, embed_dim

    def _load_glove_embedding(self, path):
        with open(path, 'r', encoding='utf8') as fp:
            # word_vectors: word --> vector
            word_vectors = {}
            for line in fp:
                line = line.strip("\n").split(" ")
                word_vectors[line[0]] = np.array(
                    list(map(float, line[1:])), dtype='float32')
        vocab_embed = word_vectors.keys()
        embed_dim = word_vectors[next(iter(vocab_embed))].shape[0]

        return word_vectors, vocab_embed, embed_dim

    def _load_bert_embedding(self, train_sents, n_label_sents, dev_sents, test_sents):
        sents = train_sents + dev_sents + test_sents

        n_sents = {'label': n_label_sents, 'dev': len(dev_sents), 'test': len(test_sents)}

        n_words = {'label': len([w for s in train_sents[:n_label_sents] for w in s]),
                   'dev': len([w for s in dev_sents for w in s]),
                   'test': len([w for s in test_sents for w in s])}

        n_unlabel_sents = len(train_sents) - n_label_sents
        if n_unlabel_sents != 0:
            n_sents['unlabel'] = n_unlabel_sents
            n_words['unlabel'] = len([w for s in train_sents[n_label_sents:] for w in s])

        if self.expe.config.embed_file:
            with open(self.expe.config.embed_file, "rb+") as infile:
                word_vectors = pickle.load(infile)

        else:
            bert_batch_size = 1
            bert_layers = '-1,-2,-3,-4'
            bert_load_features = False
            bert_max_seq_length = 512
            bert_do_lower_case = False
            bert_model = 'Musixmatch/umberto-commoncrawl-cased-v1'

            bert = Bert(bert_model, bert_layers, bert_max_seq_length, bert_batch_size, bert_do_lower_case, 0)
            bert_hidden_size = bert.model.config.hidden_size

            word_vectors = bert.extract_bert_features(sents)

            dataset_id = str(self.expe.config.data_file).rsplit('/')[-1]
            with open(f"./input/word_vectors_{dataset_id}", 'wb+') as f:
                pickle.dump(word_vectors, f, protocol=-1)

        return word_vectors, n_sents, n_words

    def _build_vocab_from_data(self, train_sents, dev_sents, test_sents):
        self.expe.log.info("vocab file not exist, start building")
        train_char_vocab = Counter()
        train_vocab = [w for s in train_sents for w in s]
        for sent in train_sents:
            for w in sent:
                for c in w:
                    train_char_vocab[c] += 1
        dev_vocab = [w for s in dev_sents for w in s]
        test_vocab = [w for s in test_sents for w in s]

        return train_char_vocab, train_vocab, dev_vocab, test_vocab

    @lazy_execute("_load_from_pickle")
    def _build_vocab_from_embedding(
            self, train_sents, n_label_sents, dev_sents, test_sents, embed_file,
            vocab_size, char_vocab_size, file_name):
        # embed_file potrebbe essere None, usando Bert
        # self.expe.log.info("loading embedding file from {}".format(embed_file))
        if self.expe.config.embed_type.lower() == "glove":
            word_vectors, vocab_embed, embed_dim = \
                self._load_glove_embedding(embed_file)
        elif self.expe.config.embed_type.lower() == "twitter":
            word_vectors, vocab_embed, embed_dim = \
                self._load_twitter_embedding(embed_file)
        elif self.expe.config.embed_type.lower() == "bert":
            word_vectors, n_sents, n_words = \
                self._load_bert_embedding(train_sents, n_label_sents, dev_sents, test_sents)
        else:
            raise NotImplementedError(f"Unsupported embedding type: {self.expe.config.embed_type}")

        train_char_vocab, train_vocab, dev_vocab, test_vocab = \
            self._build_vocab_from_data(train_sents, dev_sents, test_sents)

        word_vocab = train_vocab + dev_vocab + test_vocab

        char_ls = train_char_vocab.most_common(char_vocab_size)
        self.expe.log.info('#Chars: {}'.format(len(char_ls)))
        for key in char_ls[:5]:
            self.expe.log.info(key)
        self.expe.log.info('...')
        for key in char_ls[-5:]:
            self.expe.log.info(key)
        char_vocab = {c[0]: index + 1 for (index, c) in enumerate(char_ls)}

        char_vocab[UNK_CHAR] = UNK_CHAR_IDX

        self.expe.log.info("char vocab size: {}".format(len(char_vocab)))

        vocab = [UNK_WORD] + word_vocab
        word_vectors_pooled = torch.vstack([torch.vstack(wv) for wv in word_vectors])
        W = torch.tensor(np.random.uniform(-0.1, 0.1, size=(1, self.expe.config.edim)))
        W = torch.vstack([W, word_vectors_pooled])

        self.expe.log.info(
            "Words are initialized with loaded embeddings.")
        return W, vocab, char_vocab, n_sents, n_words

    def _load_from_pickle(self, file_name):
        self.expe.log.info("loading from {}".format(file_name))
        with open(file_name, "rb") as fp:
            data = pickle.load(fp)
        return data


class minibatcher:
    @auto_init_args
    def __init__(self, word_data, char_data, label, batch_size, shuffle):
        self._reset()

    def __len__(self):
        return len(self.idx_pool)

    def _reset(self):
        self.pointer = 0
        idx_list = np.arange(len(self.word_data))
        if self.shuffle:
            np.random.shuffle(idx_list)
        self.idx_pool = [idx_list[i: i + self.batch_size]
                         for i in range(0, len(self.word_data),
                         self.batch_size)]

    def _pad(self, word_data, char_data, labels):
        max_word_len = max([len(sent) for sent in word_data])
        max_char_len = max([len(char) for sent in char_data
                           for char in sent])

        input_data = \
            np.zeros((len(word_data), max_word_len)).astype("float32")
        input_mask = \
            np.zeros((len(word_data), max_word_len)).astype("float32")
        input_char = \
            np.zeros(
                (len(word_data), max_word_len, max_char_len)).astype("float32")
        input_char_mask = \
            np.zeros(
                (len(word_data), max_word_len, max_char_len)).astype("float32")
        input_label = \
            np.zeros((len(word_data), max_word_len)).astype("float32")

        for i, (sent, chars, label) in enumerate(
                zip(word_data, char_data, labels)):
            input_data[i, :len(sent)] = \
                np.asarray(list(sent)).astype("float32")
            input_label[i, :len(label)] = \
                np.asarray(list(label)).astype("float32")
            input_mask[i, :len(sent)] = 1.

            for k, char in enumerate(chars):
                input_char[i, k, :len(char)] = \
                    np.asarray(char).astype("float32")
                input_char_mask[i, k, :len(char)] = 1

        return [input_data, input_mask, input_char,
                input_char_mask, input_label]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.idx_pool):
            self._reset()
            raise StopIteration()

        idx = self.idx_pool[self.pointer]
        sents, chars, label = \
            self.word_data[idx], self.char_data[idx], self.label[idx]

        self.pointer += 1
        return self._pad(sents, chars, label) + [idx]
