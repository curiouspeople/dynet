import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch import nn
from Index_calculation import testclass
from Model import Model
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, mean_absolute_error, mean_squared_error, f1_score, \
    cohen_kappa_score
from torch.optim import lr_scheduler


def run_experiment(batch_size, num_epochs, learning_rate, K_fold, is_shuffle):
    batch_size = batch_size
    num_epochs = num_epochs
    learning_rate = learning_rate
    result = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 90 subject
    eigenvalues = torch.tensor(np.load("hybrid_experiment/eigenvalues.npy").real.astype(float),
                               dtype=torch.float).to(device)  # ([3600, 64, 5])
    eigenvector = torch.tensor(np.load("hybrid_experiment/eigenvector.npy"),
                               dtype=torch.float).to(device)  # (3600, 5, 2, 64, 64)
    labels = torch.tensor(np.load("hybrid_experiment/labels.npy"), dtype=torch.float).squeeze_(
        1)  # (3600)
    print(eigenvalues.shape)
    print(eigenvector.shape)
    print(labels.shape)
    # torch.Size([32400, 64, 5])
    # torch.Size([32400, 5, 2, 64, 64])
    # torch.Size([32400])

    MyDataset = TensorDataset(eigenvalues, eigenvector, labels)

    # 这个shuffle，True就是混合实验，False就是跨被试实验
    kfold = KFold(n_splits=K_fold, shuffle=is_shuffle)

    maxAcc = 0
    fold = 0

    # 记录每折最大
    maxAcc_List = []
    maxAccForAvg = []
    maxF1ForAvg = []
    maxKappaForAvg = []
    maxRecallForAvg = []
    maxPreForAvg = []
    maxMaeForAvg = []
    maxMseForAvg = []

    for train_idx, test_idx in kfold.split(MyDataset):
        fold = fold + 1
        print("fold {}....".format(fold))
        train_data = Subset(MyDataset, train_idx)
        test_data = Subset(MyDataset, test_idx)
        print("train_len:{}".format(len(train_data)))
        print("test_data:{}".format(len(test_data)))

        train_loader = DataLoader(train_data, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        model = Model().to(device)
        loss_func = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.MultiStepLR(opt, milestones=[50], gamma=0.1)

        train_acc_list = []
        train_loss_list = []
        test_acc_list = []
        test_loss_list = []
        f1_list = []
        kappa_list = []
        recall_list = []
        precision_list = []
        mae_list = []
        mse_list = []

        G = testclass()
        train_len = G.len(len(train_idx), batch_size)
        test_len = G.len(len(test_idx), batch_size)

        # 记录一个epoch最大的maxACC
        temp_max_test = 0
        max_f1 = 0
        max_recall = 0
        max_precision = 0
        max_mae = 0
        max_mse = 0
        max_kappa = 0

        for epoch in range(num_epochs):
            scheduler.step()
            # -------------------------------------------------
            total_train_correct = 0
            total_train_loss = 0
            train_data_len = 0  # 用于计算用了多少个训练的数据，计算train acc
            for data_val, data_vec, labels in train_loader:
                data_val = data_val.to(device)
                data_vec = data_vec.to(device)
                labels = labels.to(device)
                train_data_len += len(data_val)
                output = model(data_val, data_vec)
                train_loss = loss_func(output, labels.long())

                opt.zero_grad()
                train_loss.backward()
                opt.step()

                train_correct = (output.argmax(dim=1) == labels).sum()

                # train_loss_list.append(train_loss)
                total_train_loss = total_train_loss + train_loss.item()
                # train_acc_list.append(train_acc)
                total_train_correct += train_correct

            train_loss_list.insert(epoch, total_train_loss / train_data_len)
            train_acc_list.insert(epoch, total_train_correct.cpu().numpy() / train_data_len)

            # -------------------------------------------------
            total_test_correct = 0
            total_test_loss = 0
            test_data_len = 0
            with torch.no_grad():
                y_pre = []
                y_true = []
                for data_val, data_vec, labels in test_loader:
                    data_val = data_val.to(device)
                    data_vec = data_vec.to(device)
                    labels = labels.to(device)
                    test_data_len += len(data_val)
                    output = model(data_val, data_vec)
                    test_loss = loss_func(output, labels.long())

                    test_correct = (output.argmax(dim=1) == labels).sum()
                    tmp = output.argmax(dim=1).cpu().numpy()
                    y_pre.append(tmp)
                    y_true.append(labels.cpu().numpy())
                    # test_loss_list.append(test_loss)
                    total_test_loss = total_test_loss + test_loss.item()

                    # test_acc_list.append(test_acc)
                    total_test_correct += test_correct
            test_loss_list.insert(epoch, (total_test_loss / test_data_len))
            # 当前准确率
            test_acc = total_test_correct.cpu().numpy() / test_data_len
            test_acc_list.insert(epoch, test_acc)
            y_pre = np.concatenate(y_pre)
            y_true = np.concatenate(y_true)
            f1 = f1_score(y_true, y_pre, average='macro')
            f1_list.insert(epoch, f1)
            # 计算kappa
            kappa = cohen_kappa_score(y_true, y_pre, weights='quadratic')
            kappa_list.insert(epoch, kappa)
            # 计算召回率（Recall）
            recall = recall_score(y_true, y_pre, average='macro')
            recall_list.insert(epoch, recall)
            # 计算精确率（Precision）
            precision = precision_score(y_true, y_pre, average='macro')
            precision_list.insert(epoch, precision)
            # 计算平均绝对误差（MAE）
            mae = mean_absolute_error(y_true, y_pre)
            mae_list.insert(epoch, mae)
            # 计算均方误差（MSE）
            mse = mean_squared_error(y_true, y_pre)
            mse_list.insert(epoch, mse)

            if temp_max_test < test_acc:
                temp_max_test = test_acc
                max_f1 = f1
                max_kappa = kappa
                max_recall = recall
                max_precision = precision
                max_mae = mae
                max_mse = mse
                # print("save....maxAcc{:.4f} .pkl/y_pre/y_true files".format(maxAcc))
                torch.save(model.state_dict(), "./model_weight_fold/fold_kuabeishi_{}.pkl".format(fold))
                file_path = "./model_weight_fold/fold_{}_maxAcc_y_true.npy".format(fold)
                np.save(file_path, y_true)
                file_path = "./model_weight_fold/fold_{}_maxAcc_y_pre.npy".format(fold)
                np.save(file_path, y_pre)

            # print result
            # print("Epoch: {}/{} ".format(epoch + 1, num_epochs),
            #       "Training Loss: {:.4f} ".format(train_loss_list[epoch]),
            #       "Training Accuracy: {:.4f} ".format(train_acc_list[epoch]),
            #       "Test Loss: {:.4f} ".format(test_loss_list[epoch]),
            #       "test- acc: {:.4f}".format(test_acc_list[epoch]),
            #       "f1: {:.4f}".format(f1_list[epoch]),
            #       "kappa: {:.4f}".format(kappa_list[epoch]),
            #       "recall: {:.4f}".format(recall_list[epoch]),
            #       "precision: {:.4f}".format(precision_list[epoch]),
            #       "mae: {:.4f}".format(mae_list[epoch]),
            #       "mse: {:.4f}".format(mse_list[epoch])
            #       )
        result.append(temp_max_test)
        maxAccForAvg.append(temp_max_test)
        maxF1ForAvg.append(max_f1)
        maxKappaForAvg.append(max_kappa)
        maxRecallForAvg.append(max_recall)
        maxPreForAvg.append(max_precision)
        maxMaeForAvg.append(max_mae)
        maxMseForAvg.append(max_mse)
        maxAcc_List.append("kFold: {}/5 ".format(
            fold) + "Test MAX Accuracy:{:.4f} f1:{:.4f} kappa:{:.4f} recall:{:.4f} precision:{:.4f} mae:{:.4f} mse:{:.4f}".format(
            temp_max_test, max_f1, max_kappa,
            max_recall,
            max_precision,
            max_mae, max_mse))
        line = "kFold: {}/5 ".format(
            fold) + "Test MAX Accuracy:{:.4f} f1:{:.4f} kaapa:{:.4f} recall:{:.4f} precision:{:.4f} mae:{:.4f} mse:{:.4f}".format(
            temp_max_test, max_f1, max_kappa,
            max_recall,
            max_precision,
            max_mae, max_mse)
        f = open("./figure/maxAcc.txt", "a")
        f.write(line + '\n')
        f.close()
        print(maxAcc_List)

        # 画图loss
        # print(len(test_loss_list))
        # print(len(test_loss_list))
        # plt.figure()
        # ax = plt.axes()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        #
        # plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, linewidth=1, linestyle='solid', label='train_loss')
        # plt.plot(range(1, len(test_loss_list) + 1), test_loss_list, linewidth=1, linestyle='solid', label='eval_loss')
        #
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.legend()
        # plt.title('Loss curve')
        # plt.savefig('./figure/loss_fold.png')

        # train_x1 = range(0, num_epochs)
        # train_x2 = range(0, num_epochs)
        # train_y1 = train_acc_list
        # train_y2 = train_loss_list
        # test_x1 = range(0, num_epochs)
        # test_x2 = range(0, num_epochs)
        # test_y1 = test_acc_list
        # test_y2 = test_loss_list
        # plt.subplot(2, 1, 1)
        # plt.plot(train_x1, train_y1, alpha=0.7, linewidth=1, color="blue", label='trainAcc')
        # # plt.plot(train_x1, train_y1, 'bv--', alpha=0.7, linewidth=2, label='ESTCNN', color="blue")
        # plt.plot(test_x1, test_y1, alpha=0.7, linewidth=1, color="red", label='testAcc')
        # plt.title("{} accuracy vs. epoches".format(fold))
        # plt.ylabel('accuracy')
        # plt.legend()
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(train_x2, train_y2, alpha=0.7, linewidth=1, color="blue", label='trainLoss')
        # plt.title("{} loss vs. epoches".format(fold))
        # plt.plot(test_x2, test_y2, alpha=0.7, linewidth=1, color="red", label='testLoss')
        # plt.ylabel('loss')
        # plt.legend()
        # plt.subplots_adjust(wspace=0.3, hspace=0.4)
        # plt.show()
        # plt.savefig('./figure/fold_kuabeishi_{}.png'.format(fold))
        # plt.close()
    line = "Avg acc:{:.4f}, f1:{:.4f} kappa:{:.4f} recall:{:.4f} precision:{:.4f} mae:{:.4f} mse:{:.4f}".format(
        sum(maxAccForAvg) / len(maxAccForAvg),
        sum(maxF1ForAvg) / len(maxF1ForAvg),
        sum(maxKappaForAvg) / len(maxKappaForAvg),
        sum(maxRecallForAvg) / len(maxRecallForAvg),
        sum(maxPreForAvg) / len(maxPreForAvg),
        sum(maxMaeForAvg) / len(maxMaeForAvg),
        sum(maxMseForAvg) / len(maxMseForAvg)
    )
    f = open("./figure/maxAcc.txt", "a")
    paramter = "paramter: batch size:{}, epoch:{}, learning rate:{}, K_fold:{}, is_shuffle:{}".format(batch_size,
                                                                                                      num_epochs,
                                                                                                      learning_rate,
                                                                                                      K_fold,
                                                                                                      is_shuffle)
    f.write(paramter + '\n')
    f.write(line + '\n')
    line = "---------------------------------------------------------------\n"
    f.write(line + '\n')
    f.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 统计模型参数数量
model = Model()
print(f'Total Trainable Parameters: {count_parameters(model)}')
# 这个shuffle，True就是混合实验，False就是跨被试实验
run_experiment(64, 200, 0.005, 9, is_shuffle=False)
