import matplotlib.pyplot as plt


if __name__ == '__main__':
    evalita_st = [97.40, 97.45, 97.66, 97.75]
    isdt_st = [97.60, 97.69, 97.75, 97.83]
    twita_st = [95.20, 95.40, 95.60, 95.70]
    st = {'evalita': evalita_st, 'isdt': isdt_st, 'twita': twita_st}
    x_labs = ['20% L', '+20% U', '+50% U', '+80% U']

    plt.plot(x_labs, st['evalita'], label='EVALITA', color='tab:red')
    # plt.plot(x_labs, st['isdt'], label='ISDT UD2.5', color='b')
    # plt.plot(x_labs, st['twita'], label='PoSTWITA UD2.5', color='g')
    plt.ylim([97.38,98.40])
    plt.xlabel('Unlabeled data added (+X% U)', labelpad=10, fontsize=12)
    plt.ylabel('Accuracy (%)', labelpad=10, fontsize=12)
    # plt.legend()

    # plt.plot(x_labs, st['isdt'], label='ISDT UDv2.5', color='tab:red')
    # # plt.plot(x_labs, st['isdt'], label='ISDT UD2.5', color='b')
    # # plt.plot(x_labs, st['twita'], label='PoSTWITA UD2.5', color='g')
    # plt.ylim([97.58,98.6])
    # plt.xlabel('Unlabeled data added (+X% U)', labelpad=10, fontsize=12)
    # plt.ylabel('Accuracy (%)', labelpad=10, fontsize=12)

    # plt.plot(x_labs, st['twita'], label='PoSTWITA UDv2.5', color='tab:red')
    # # plt.plot(x_labs, st['isdt'], label='ISDT UD2.5', color='b')
    # # plt.plot(x_labs, st['twita'], label='PoSTWITA UD2.5', color='g')
    # plt.ylim([95.18,96.20])
    # plt.xlabel('Unlabeled data added (+X% U)', labelpad=10, fontsize=12)
    # plt.ylabel('Accuracy (%)', labelpad=10, fontsize=12)