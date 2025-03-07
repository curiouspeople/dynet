# Author: Thel'Vadamee
# Date: 2023/11/15 10:11

import os
import datetime
import time
import matplotlib.pyplot as plt


def labels2class(y):
    classes_lst = []
    for value in y:
        if value <= 0.35:
            classes_lst.append(0)
        elif value <= 0.7:
            classes_lst.append(1)
        else:
            classes_lst.append(2)
    return classes_lst


# batch_size, n_epoch, lr, momentum, k_fold
# 每个被试完成时创建
def recorder(root, n_fold, max_train_Acc, max_eval_Acc):
    with open(root, 'a') as f:
        f.write('第{}折：\n'.format(n_fold))
        f.write('\tmax train Acc：{}\n'.format(max_train_Acc))
        f.write('\tmax eval Acc：{}\n'.format(max_eval_Acc))


# 创建文件夹(程序启动时创建)
def createdir():
    # 创建文件夹
    root = '../recording'
    timrstr = time.strftime("%Y%m%d-%H%M%S")
    print(timrstr)
    dir_name = ''
    if os.path.exists('../recording/timrstr'):
        dir_name = timrstr + '_re'
    else:
        dir_name = timrstr
    dir_root = os.path.join(root, dir_name)
    os.mkdir(dir_root)

    return dir_root


# 创建说明文件(每次启动程序时创建)
def createREADME(dir_root, batch_size, n_epoch, lr, momentum, k_fold):
    with open(os.path.join(dir_root, 'README.txt'), 'w') as f:
        f.write('batch_size = {}\n'
                'n_epoch={}\n'
                'lr={}\n'
                'k_flod={}\n'
                'momentum={}'.format(str(batch_size), str(n_epoch), str(lr), str(k_fold), str(momentum)))


def drawloss(subject, fold, train_loss, eval_loss):
    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(range(1, len(train_loss) + 1), train_loss, linewidth=1, linestyle='solid', label='train_loss')
    plt.plot(range(1, len(eval_loss) + 1), eval_loss, linewidth=1, linestyle='solid', label='eval_loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss curve')
    plt.savefig('../../figure/loss_{}_{}.png'.format(subject, fold))
    # plt.show()


def drawacc(subject, fold, train_acc_lst, eval_acc_lst):
    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(range(1, len(train_acc_lst) + 1), train_acc_lst, linewidth=1, linestyle='solid', label='train_acc')
    plt.plot(range(1, len(eval_acc_lst) + 1), eval_acc_lst, linewidth=1, linestyle='solid', label='eval_acc')

    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.title('Acc curve')
    plt.savefig('../../figure/acc_{}_{}.png'.format(subject, fold))
    # plt.show()
#
# if __name__ == '__main__':

