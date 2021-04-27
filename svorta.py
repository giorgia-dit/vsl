import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    datasets = ['ISDT (UD 2.5)\nBaseline: 97.0%', 'PoSTWITA (UD 2.5)\nBaseline: 93.9%', 'Evalita 2007\nBaseline: 96.2%']
    base = [97.0, 93.9, 96.2]
    # sd_base = [0.09, 0.01, 0.10]
    # sd_bert = [0.10, 0.02, 0.04]
    # sd_bert_ul = [0.06, 0.09, 0.06]
    # base_ul = [97.1, 94.1, 96.6]
    base_ul = [0.1, 0.2, 0.4]
    bert_abs = [97.6, 95.2, 97.4]
    bert = [0.6, 1.3, 1.2]
    # bert_ul = [97.8, 95.7, 97.7]
    bert_ul = [0.8, 1.8, 1.5]

    data_lab = np.arange(len(datasets))
    width = 0.35

    plt.style.use('seaborn-dark')

    fig, (ax) = plt.subplots()

    rects1 = ax.bar(data_lab - width / 2, base_ul, color='#9AB7B8', width=width, label='Baseline + unlabeled data')
    # rects1 = ax.bar(data_lab - width / 2, base_ul, color='#fcb16d', width=width, label='Baseline + unlabeled data')
    rects3 = ax.bar(data_lab + width / 2, bert_ul, color='#46879e', width=width, label='BERT-based model + unlabeled data')
    # rects3 = ax.bar(data_lab + width / 2, bert_ul, color='#468d96', width=width, label='Bert-based model + unlabeled data')
    rects2 = ax.bar(data_lab + width / 2, bert, color='#68c2ca', width=width, label='BERT-based model')
    # rects2 = ax.bar(data_lab + width / 2, bert, color='#81c4ca', width=width, label='Bert-based model')

    ax.set_ylim((0,2))
    ax.set_xticks(data_lab)
    ax.set_xticklabels(datasets, fontsize=15)
    ax.set_yticklabels(ax.get_yticks(), fontsize=14)
    ax.set_ylabel('Average accuracy increase (%)', fontsize=15, labelpad=10)

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels = [labels[0], labels[2], labels[1]]
    handles = [handles[0], handles[2], handles[1]]
    ax.legend(handles, labels, fontsize=15, loc='upper left')



    def autolabel(rects, baseline):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(f'{height + baseline[i]:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points", fontsize=13, weight='bold',
                        ha='center', va='bottom')

    autolabel(rects1, base)
    autolabel(rects2, base)
    autolabel(rects3, base)


    plt.title('Model accuracy variation over baseline [labeled data: 20% corpus – unlabeled data: 80% corpus]', fontsize=16, weight='bold', pad=15)
    plt.show()