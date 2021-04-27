import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


if __name__ == '__main__':
    evalita_st = [97.40, 97.45, 97.66, 97.75]
    isdt_st = [97.60, 97.69, 97.75, 97.83]
    twita_st = [95.20, 95.40, 95.60, 95.70]
    st = {'evalita': evalita_st, 'isdt': isdt_st, 'twita': twita_st}
    x_labs = ['20% L', '+20% U', '+50% U', '+80% U']
    #
    # plt.clf()
    # plt.style.use('seaborn-dark')
    # plt.plot(x_labs, st['evalita'], label='EVALITA', color='#46879e')
    # # plt.plot(x_labs, st['isdt'], label='ISDT UD2.5', color='b')
    # # plt.plot(x_labs, st['twita'], label='PoSTWITA UD2.5', color='g')
    # plt.ylim([97.38,98.40])
    # plt.xlabel('Unlabeled data added (+X% U)', labelpad=10, fontsize=15)
    # plt.ylabel('Accuracy (%)', labelpad=10, fontsize=15)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=14)
    # # plt.title('Quantity of unlabeled samples effect on model accuracy', fontsize=16, weight='bold', pad=15)
    # plt.show()
    # # plt.legend()
    #
    # plt.clf()
    # plt.style.use('seaborn-dark')
    # plt.plot(x_labs, st['isdt'], label='ISDT UDv2.5', color='#46879e')
    # # plt.plot(x_labs, st['isdt'], label='ISDT UD2.5', color='b')
    # # plt.plot(x_labs, st['twita'], label='PoSTWITA UD2.5', color='g')
    # plt.ylim([97.58, 98.6])
    # plt.xlabel('Unlabeled data added (+X% U)', labelpad=10, fontsize=15)
    # plt.ylabel('Accuracy (%)', labelpad=10, fontsize=15)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=14)
    # # plt.title('Quantity of unlabeled samples effect on model accuracy', fontsize=16, weight='bold', pad=15)
    # plt.show()
    #
    plt.clf()
    plt.style.use('seaborn-dark')
    plt.plot(x_labs, st['twita'], label='PoSTWITA UDv2.5', color='#46879e')
    # plt.plot(x_labs, st['isdt'], label='ISDT UD2.5', color='b')
    # plt.plot(x_labs, st['twita'], label='PoSTWITA UD2.5', color='g')
    plt.ylim([95.18,96.20])
    plt.xlabel('Unlabeled data added (+X% U)', labelpad=10, fontsize=15)
    plt.ylabel('Accuracy (%)', labelpad=10, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=14)
    plt.title('Quantity of unlabeled samples effect on model accuracy',
              fontsize=16, weight='bold', pad=30)
    plt.show()

    # PLOT 2
    # evalita_st_100 = [98.58, 98.62, 98.65]
    # evalita_st_50 = [98.22, 98.25, 98.30]
    # isdt_st_100 = []
    # isdt_st_50 = [98.15, 98.2]
    # twita_st_100 = [96.80, 96.82, 96.83]
    # twita_st_50 = [96.22, 96.25, 96.28]
    # st = {'evalita_100': evalita_st_100, 'isdt_100': isdt_st_100, 'twita_100': twita_st_100,
    #       'evalita_50': evalita_st_50, 'isdt_50': isdt_st_50, 'twita_50': twita_st_50}
    # x_labs = ['Base', '+50% U', '+100% U']

    # plt.clf()
    # plt.style.use('seaborn-dark')
    # plt.plot(['Base', '+50% U'], st['isdt_50'], label='Base = 50% L')
    # #plt.plot(x_labs, st['isdt_100'], label='Base = 100% L', color='tab:blue')
    # # plt.plot(x_labs, st['isdt'], label='ISDT UD2.5', color='b')
    # # plt.plot(x_labs, st['twita'], label='PoSTWITA UD2.5', color='g')
    # plt.xlabel('Unlabeled data added (+X% U)', labelpad=10, fontsize=12)
    # plt.ylabel('Accuracy (%)', labelpad=10, fontsize=12)
    # plt.ylim([98.10, 99.1])
    # plt.legend()
    # plt.show()

    # plt.clf()
    # plt.style.use('seaborn-dark')
    # plt.plot(x_labs, st['twita_50'], label='Base = 50% L')
    # plt.plot(x_labs, st['twita_100'], label='Base = 100% L')
    # plt.ylim([96.18, 97.2])
    # plt.xlabel('Unlabeled data added (+X% U)', labelpad=10, fontsize=12)
    # plt.ylabel('Accuracy (%)', labelpad=10, fontsize=12)
    # plt.legend()
    # plt.show()

    # plt.clf()
    # plt.style.use('seaborn-dark')
    # plt.plot(x_labs, st['evalita_50'], label='Base = 50% L')
    # plt.plot(x_labs, st['evalita_100'], label='Base = 100% L')
    # plt.ylim([98.18, 99.2])
    # plt.xlabel('Unlabeled data added (+X% U)', labelpad=10, fontsize=12)
    # plt.ylabel('Accuracy (%)', labelpad=10, fontsize=12)
    # plt.legend()
    # plt.show()
