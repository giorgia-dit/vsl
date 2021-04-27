import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


# use grouped == True for EVALITA ONLY
def build_tag_vocab(data_dictionary, grouped=False):
    df = pd.DataFrame.from_dict(data_dictionary, orient='index')
    tag_vocabulary = set(df['true_tag'])
    if '_' in tag_vocabulary:
        tag_vocabulary.remove('_')
    if 'X' in tag_vocabulary:
        tag_vocabulary.remove('X')
    if 'NULL' in tag_vocabulary:
        tag_vocabulary.remove('NULL')
    grouped_tag_vocabulary = None
    if grouped:
        grouped_tag_vocabulary = {
            'V': [], 'NN': [], 'NN_P': [], 'ART': [], 'PREP': [], 'ADJ': [],
            'CONJ': [], 'ADV': [], 'INT': [], 'C_NUM': [], 'PRON': [], 'P': []
        }
        for tag in tag_vocabulary:
            for group in grouped_tag_vocabulary.keys():
                if tag.startswith(group):
                    grouped_tag_vocabulary[group].append(tag)
        for tag in tag_vocabulary:
            if tag.startswith('PR'):
                grouped_tag_vocabulary['P'].remove(tag)
        grouped_tag_vocabulary['NN'].remove('NN_P')
    return tag_vocabulary, grouped_tag_vocabulary


# tag_vocabulary can be grouped (type: dictionary), or fine-grained (type: set)
def pred_count(data_dictionary, tag_vocabulary, grouped=False):
    cnt = {}
    if not grouped:
        for tag in tag_vocabulary:
            cnt[tag] = {'right': 0, 'wrong': 0}
            for obs in data_dictionary.keys():
                if data_dictionary[obs]['true_tag'] == tag:
                    if data_dictionary[obs]['res'] == 0:
                        cnt[tag]['wrong'] += 1
                    elif data_dictionary[obs]['res'] == 1:
                        cnt[tag]['right'] += 1
    else:
        for group in tag_vocabulary.keys():
            cnt[group] = {'right': 0, 'wrong': 0}
            for tag in tag_vocabulary[group]:
                for obs in data_dictionary.keys():
                    if data_dictionary[obs]['true_tag'] == tag:
                        if data_dictionary[obs]['res'] == 0:
                            cnt[group]['wrong'] += 1
                        elif data_dictionary[obs]['res'] == 1:
                            cnt[group]['right'] += 1

    total = 0
    total_err = 0
    for k in cnt.keys():
        total_err += cnt[k]['wrong']
        total_occ = cnt[k]['right'] + cnt[k]['wrong']
        total += total_occ

    for k in cnt.keys():
        total_occ = cnt[k]['right'] + cnt[k]['wrong']
        total_err_occ = cnt[k]['wrong']
        cnt[k]['err_rel_perc'] = round(cnt[k]['wrong'] * 100 / total_err, 2)
        cnt[k]['err_occ_perc'] = round(cnt[k]['wrong'] * 100 / total_occ, 2)
        cnt[k]['total_occ'] = total_occ
        cnt[k]['total_err_occ'] = total_err_occ
        cnt[k]['perc_occ'] = cnt[k]['total_occ'] * 100 / total

    return cnt

def combination_count(data_dictionary, tag_vocabulary, grouped=False):
    comb_count = {}
    if not grouped:
        for tag in sorted(list(tag_vocabulary)):
            comb_count[tag] = {}
            for tag_prime in sorted(list(tag_vocabulary)):
                if tag_prime != tag:
                    comb_count[tag][tag_prime] = 0
                    for obs in data_dictionary.keys():
                        if data_dictionary[obs]['true_tag'] == tag:
                            if data_dictionary[obs]['res'] == 0:
                                if data_dictionary[obs]['pred_tag'] == tag_prime:
                                    comb_count[tag][tag_prime] += 1
    else:
        for group in sorted(tag_vocabulary.keys()):
            comb_count[group] = {}
            for group_prime in sorted(tag_vocabulary.keys()):
                if group_prime != group:
                    comb_count[group][group_prime] = 0
                    for tag in sorted(tag_vocabulary[group]):
                        for obs in data_dictionary.keys():
                            if data_dictionary[obs]['true_tag'] == tag:
                                if data_dictionary[obs]['res'] == 0:
                                    for tag_prime in sorted(tag_vocabulary[group_prime]):
                                        if data_dictionary[obs]['pred_tag'] == tag_prime:
                                            comb_count[group][group_prime] += 1

    return comb_count

# bar plot with err percentages for tag on total occ
def plot_err_perc(count_dict, dataset):
    plt.style.use('seaborn-dark')
    four = [(tag, count_dict[tag]['err_occ_perc'], count_dict[tag]['total_occ']) for tag in count_dict.keys()]
    four.sort(key=lambda x: x[1])
    plt.bar([p[0] for p in four], [p[1] for p in four], width=0.5)
    plt.ylabel('Wrong predictions (%) on total tag occurrences', labelpad=10, fontsize=12)
    if dataset == 'evalita':
        plt.xlabel('Tag group', labelpad=10, fontsize=12)
    else:
        plt.xlabel('Tag', labelpad=10, fontsize=12)
    if dataset == 'twita':
        plt.ylim((0, 15))
    else:
        plt.ylim((0, 10))
    for (tag, perc, total_occ) in four:
        plt.text(tag, perc + 0.4, f"{perc}%\n({total_occ})", horizontalalignment='center',
                 verticalalignment='center', fontsize=9.5)

    plt.tight_layout(pad=0)
    plt.show()
    # plt.savefig(f"./{dataset}_plot1.pdf", bbox_inches='tight', pad_inches=0.4)

def plot_err_rel(count_dict, dataset):
    plt.clf()
    plt.style.use('seaborn-dark')
    pairs = [(tag, count_dict[tag]['err_rel_perc']) for tag in count_dict.keys()]
    pairs.sort(key=lambda x: x[1])
    plt.bar([p[0] for p in pairs], [p[1] for p in pairs], width=0.5, color='xkcd:deep orange')
    plt.ylabel('Wrong predictions (%) for each tag on total wrong predictions',
               labelpad=10, fontsize=12)
    if dataset == 'evalita':
        plt.xlabel('Tag group', labelpad=10, fontsize=12)
    else:
        plt.xlabel('Tag', labelpad=10, fontsize=12)
    plt.ylim((0, 30))
    for (tag, perc) in pairs:
        plt.text(tag, perc + 0.5, f"{perc}%", horizontalalignment='center',
                 verticalalignment='center', fontsize=9.5)

    plt.tight_layout(pad=0)
    plt.show()

def plot_tag_dist(count_dict, train_dict, dataset):
    plt.clf()
    plt.style.use('seaborn-dark')
    triples = [(tag, count_dict[tag]['perc_occ'], train_dict[tag]) for tag in count_dict.keys()]
    triples.sort(key=lambda x: x[1])
    print(triples)
    _X = np.arange(len([t[0] for t in triples]))
    plt.bar(_X -0.2, [p[1] for p in triples], width=0.35, label='Test set', color='#3B75AF')
    plt.bar(_X +0.2, [p[2] for p in triples], width=0.35, label='Training set', color='lightsteelblue')
    plt.xticks(_X, [t[0] for t in triples], fontsize=10)
    plt.ylabel('Distribution of tag occurrences (%) in test and training set', labelpad=10, fontsize=12)
    if dataset == 'evalita':
        plt.xlabel('Tag group', labelpad=10, fontsize=12)
    else:
        plt.xlabel('Tag', labelpad=10, fontsize=12)
    plt.ylim((0, 25))
    plt.legend()
    plt.tight_layout(pad=0)
    plt.show()

def plot_combinations(comb_count, dataset):
    tot_4tag = {}
    tot = 0
    for k in comb_count.keys():
        tot_4tag[k] = 0
        for t in comb_count[k].keys():
            tot_4tag[k] += comb_count[k][t]
    for k in tot_4tag.keys():
        tot += tot_4tag[k]

    check_count = 0
    for k in comb_count.keys():
        for t in comb_count[k].keys():
            # if tot[k] != 0:
            #     comb_count[k][t] = round(comb_count[k][t] * tot[k], 1)
            c = comb_count[k][t] * 100 / tot  # perc ponderata
            comb_count[k][t] = round(c, 1)
            check_count += comb_count[k][t]

    print(check_count)

    array = np.zeros((len(comb_count.keys()), len(comb_count.keys())))
    for i in range(len(comb_count.keys())):
        array[i][i] = 0
        for j in range(len(comb_count.keys())):
            if j != i:
                array[i][j] = comb_count[list(comb_count.keys())[i]][list(comb_count.keys())[j]]

    plt.style.use('seaborn-dark')

    fig, ax = plt.subplots()
    im = ax.imshow(array, cmap='Oranges')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Relative frequency (%) of true-predicted tags combination', fontsize=12, rotation=-90, va="bottom")

    plt.xlabel('Predicted tag', labelpad=10, fontsize=12)
    plt.ylabel('True tag', labelpad=10, fontsize=12)

    ax.set_xticks(np.arange(len(comb_count.keys())))
    ax.set_yticks(np.arange(len(comb_count.keys())))

    ax.set_xticklabels(list(comb_count.keys()), fontsize=10, minor=False)
    ax.set_yticklabels(list(comb_count.keys()), fontsize=10, minor=False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(comb_count.keys())):
        ax.text(i, i, 0, ha="center", va="center", color="w")
        for j in range(len(comb_count.keys())):
            if j != i:
                ax.text(j, i, comb_count[list(comb_count.keys())[i]][list(comb_count.keys())[j]],
                        ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()

def nemar_test(data_dict1, data_dict2):
    nemar_dict = {}
    for obs_i in range(len(data_dict1.keys())):
        if data_dict1[list(data_dict1.keys())[obs_i]]['true_tag'] != data_dict2[list(data_dict2.keys())[obs_i]]['true_tag']:
            print(list(data_dict1.keys())[obs_i], list(data_dict2.keys())[obs_i], ' Error!')
        nemar_dict[obs_i] = {}
        nemar_dict[obs_i]['c1'] = data_dict1[list(data_dict1.keys())[obs_i]]['res']
        nemar_dict[obs_i]['c2'] = data_dict2[list(data_dict2.keys())[obs_i]]['res']

    table = [[0, 0], [0, 0]]

    for obs_i in range(len(nemar_dict.keys())):
        if nemar_dict[obs_i]['c1'] == nemar_dict[obs_i]['c2']:
            if nemar_dict[obs_i]['c1'] == 1:
                table[0][0] += 1
            else:
                table[1][1] += 1
        else:
            if nemar_dict[obs_i]['c1'] == 1:
                table[0][1] += 1
            else:
                table[1][0] += 1

    result = mcnemar(table, exact=False, correction=True)

    return table, result



if __name__ == '__main__':
    dataset = ''

    if dataset == 'evalita':
        file_name = './' + 'output_evalita_100_nemar/' + 'nemar_29999.pkl'
        with open(file_name, "rb") as fp:
            evalita_data = pickle.load(fp)

        evalita_tag_voc, evalita_grouped_tag_voc = build_tag_vocab(evalita_data, grouped=True)

        evalita_count = pred_count(evalita_data, evalita_grouped_tag_voc, grouped=True)

        evalita_train_data = {
            'V': 14.44, 'NN': 19.90, 'NN_P': 4.22, 'ART': 8.50, 'PREP': 15.45, 'ADJ': 9.42,
            'CONJ': 5.47, 'ADV': 5.15, 'INT': 0.02, 'C_NUM': 0.81, 'PRON': 4.78, 'P': 11.95
        }

        evalita_comb_count = combination_count(evalita_data, evalita_grouped_tag_voc, grouped=True)

        # plot_err_perc(evalita_count, dataset=dataset)

        # plot_err_rel(evalita_count,dataset=dataset)

        # plot_tag_dist(evalita_count, evalita_train_data, 'evalita')

        # plot_combinations(evalita_comb_count,dataset=dataset)



    elif dataset == 'isdt':
        file_name = './' + 'output_isdt_100_nemar/' + 'nemar_29999.pkl'
        with open(file_name, "rb") as fp:
            isdt_data = pickle.load(fp)

        isdt_tag_voc, _ = build_tag_vocab(isdt_data)

        isdt_count = pred_count(isdt_data, isdt_tag_voc)

        isdt_train_data = {
            'ADJ': 6.63,
            'ADP': 15.12,
            'ADV': 3.87,
            'AUX': 4.01,
            'CCONJ': 2.70,
            'DET': 16.32,
            'INTJ': 0.02,
            'NOUN': 19.88,
            'NUM': 1.70,
            'PART': 0.01,
            'PRON': 3.81,
            'PROPN': 5.01,
            'PUNCT': 11.36,
            'SCONJ': 1.03,
            'SYM': 0.03,
            'VERB': 8.52
        }

        isdt_comb_count = combination_count(isdt_data, isdt_tag_voc, grouped=False)


        # plot_err_perc(isdt_count, dataset=dataset)

        # plot_err_rel(isdt_count,dataset=dataset)

        # plot_tag_dist(isdt_count, isdt_train_data, 'isdt')

        # plot_combinations(isdt_comb_count, dataset=dataset)



    elif dataset == 'twita':
        file_name = './' + 'output_postwita_100_nemar/' + 'nemar_19999.pkl'
        with open(file_name, "rb") as fp:
            twita_data = pickle.load(fp)

        twita_tag_voc, _ = build_tag_vocab(twita_data)

        twita_count = pred_count(twita_data, twita_tag_voc)

        twita_train_data = {
            'ADJ': 4.03,
            'ADP': 10.58,
            'ADV': 5.22,
            'AUX': 3.61,
            'CCONJ': 2.39,
            'DET': 11.72,
            'INTJ': 1.13,
            'NOUN': 14.41,
            'NUM': 1.10,
            'PRON': 5.28,
            'PROPN': 8.44,
            'PUNCT': 12.13,
            'SCONJ': 1.27,
            'SYM': 9.56,
            'VERB': 9.13
        }

        twita_comb_count = combination_count(twita_data, twita_tag_voc, grouped=False)


        # plot_err_perc(twita_count, dataset=dataset)

        # plot_err_rel(twita_count,dataset=dataset)

        # plot_tag_dist(twita_count,twita_train_data,'twita')

        # plot_combinations(twita_comb_count, dataset=dataset)


    else:
        # EVALITA

        # file_name = './' + 'output_evalita_20_nemar/' + 'nemar_9999.pkl'
        # with open(file_name, "rb") as fp:
        #     evalita20_data = pickle.load(fp)
        #
        # file_name = './' + 'output_evalita_20_80_nemar/' + 'nemar_9999.pkl'
        # with open(file_name, "rb") as file:
        #     evalita2080_data = pickle.load(file)
        #
        #
        # table, result = nemar_test(evalita20_data, evalita2080_data)
        # print(table)
        #
        # # summarize the finding
        # print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
        # # interpret the p-value
        # alpha = 0.05
        # if result.pvalue > alpha:
        #     print('Same proportions of errors (fail to reject H0)')
        # else:
        #     print('Different proportions of errors (reject H0)')

        # POSTWITA

        # file_name = './' + 'output_postwita_20_nemar/' + 'nemar_9999.pkl'
        # with open(file_name, "rb") as fp:
        #     twita20_data = pickle.load(fp)
        #
        # file_name = './' + 'output_postwita_20_80_nemar/' + 'nemar_9999.pkl'
        # with open(file_name, "rb") as file:
        #     twita2080_data = pickle.load(file)
        #
        # assert len(twita20_data.keys()) == len(twita2080_data.keys())
        #
        # table, result = nemar_test(twita20_data, twita2080_data)
        # print(table)
        #
        # # summarize the finding
        # print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
        # # interpret the p-value
        # alpha = 0.05
        # if result.pvalue > alpha:
        #     print('Same proportions of errors (fail to reject H0)')
        # else:
        #     print('Different proportions of errors (reject H0)')

        # ISDT

        # file_name = './' + 'output_isdt_20_nemar/' + 'nemar_29999.pkl'
        # with open(file_name, "rb") as fp:
        #     isdt20_data = pickle.load(fp)
        #
        # file_name = './' + 'output_isdt_20_80_nemar/' + 'nemar_29999.pkl'
        # with open(file_name, "rb") as file:
        #     isdt2080_data = pickle.load(file)
        #
        # assert len(isdt20_data.keys()) == len(isdt2080_data.keys())
        #
        # table, result = nemar_test(isdt20_data, isdt2080_data)
        # print(table)
        #
        # # summarize the finding
        # print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
        # # interpret the p-value
        # alpha = 0.05
        # if result.pvalue > alpha:
        #     print('Same proportions of errors (fail to reject H0)')
        # else:
        #     print('Different proportions of errors (reject H0)')


        # EVALITA BERT

        # file_name = './' + 'output_evalita_100_BASE_nemar/' + 'nemar_29999.pkl'
        # with open(file_name, "rb") as fp:
        #     evalita_base_data = pickle.load(fp)
        #
        # file_name = './' + 'output_evalita_100_nemar/' + 'nemar_29999.pkl'
        # with open(file_name, "rb") as file:
        #     evalita_bert_data = pickle.load(file)
        #
        # del evalita_bert_data[0.0]
        #
        # assert len(evalita_base_data.keys()) == len(evalita_bert_data.keys())
        #
        # table, result = nemar_test(evalita_base_data, evalita_bert_data)
        # print(table)
        #
        # # summarize the finding
        # print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
        # # interpret the p-value
        # alpha = 0.05
        # if result.pvalue > alpha:
        #     print('Same proportions of errors (fail to reject H0)')
        # else:
        #     print('Different proportions of errors (reject H0)')


        # TWITA BERT

        # file_name = './' + 'output_postwita_100_BASE_nemar/' + 'nemar_29999.pkl'
        # with open(file_name, "rb") as fp:
        #     postwita_base_data = pickle.load(fp)
        #
        # file_name = './' + 'output_postwita_100_nemar/' + 'nemar_19999.pkl'
        # with open(file_name, "rb") as file:
        #     postwita_bert_data = pickle.load(file)
        #
        # del postwita_bert_data[0.0]
        #
        # assert len(postwita_base_data.keys()) == len(postwita_bert_data.keys())
        #
        #
        # table, result = nemar_test(postwita_base_data, postwita_bert_data)
        # print(table)
        #
        # # summarize the finding
        # print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
        # # interpret the p-value
        # alpha = 0.05
        # if result.pvalue > alpha:
        #     print('Same proportions of errors (fail to reject H0)')
        # else:
        #     print('Different proportions of errors (reject H0)')


        # ISDT BERT

        file_name = './' + 'output_isdt_100_BASE_nemar/' + 'nemar_29999.pkl'
        with open(file_name, "rb") as fp:
            isdt_base_data = pickle.load(fp)

        file_name = './' + 'output_isdt_100_nemar/' + 'nemar_19999.pkl'
        with open(file_name, "rb") as file:
            isdt_bert_data = pickle.load(file)

        del isdt_bert_data[0.0]

        assert len(isdt_base_data.keys()) == len(isdt_bert_data.keys())

        table, result = nemar_test(isdt_base_data, isdt_bert_data)
        print(table)

        # summarize the finding
        print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
        # interpret the p-value
        alpha = 0.05
        if result.pvalue > alpha:
            print('Same proportions of errors (fail to reject H0)')
        else:
            print('Different proportions of errors (reject H0)')



