import numpy as np
from scipy.signal import butter, lfilter


#  假设采样频率为400hz,信号本身最大的频率为200hz，要滤除0.5hz以下，50hz以上频率成分，即截至频率为0.5hz，50hz
def butter_bandpass_filter(data, lowcut, highcut, samplingRate, order=5):
    nyq = 0.5 * samplingRate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


# 特征向量归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# -------------------------------------------------------------------------
# calculate pcc
# i = 1
# for i in range(1, 91):
#     data = np.load(f"D:/Code/Data/PhysioNet/raw_data/S{i}.npy") # (57600, 64)
#     labels = np.load(f"D:/Code/Data/PhysioNet/label/S{i}.npy") # (360, 1)
#
#     channels = data.shape[1]
#     frequency = 160
#
#     data_list = []
#     for channel in range(channels):
#
#         trail_single = data[:, channel]
#
#         Delta = butter_bandpass_filter(trail_single, 0.5, 4, frequency, order=3)
#         Theta = butter_bandpass_filter(trail_single, 4, 8, frequency, order=3)
#         Alpha = butter_bandpass_filter(trail_single, 8, 12, frequency, order=3)
#         Beta = butter_bandpass_filter(trail_single, 12, 30, frequency, order=3)
#         Gamma = butter_bandpass_filter(trail_single, 30, 50, frequency, order=3)
#
#         Fre_list = [Delta, Theta, Alpha, Beta, Gamma]
#         data_list.append(Fre_list)
#     #     print(np.array(Fre_list).shape)
#     # print(np.array(data_list).shape)
#
#     data = np.array(data_list).transpose(2, 0, 1)
#     vsplit_data = np.vsplit(data, labels.shape[0]) # (360, 160, 64, 5)
#
#     cor_list = []
#     for vsplit_datum in vsplit_data:
#         vsplit_datum = vsplit_datum.transpose(2, 1, 0)
#         # print(vsplit_datum.shape)
#         f_list = []
#         for c in range(5):
#             cor = np.corrcoef(vsplit_datum[c, :, :])
#             f_list.append(cor)
#         cor_list.append(f_list)
#
#     corr = np.array(cor_list)
#     print(corr.shape)
#     np.save(f"D:/Code/Data/PhysioNet/cor/S{i}.npy", corr)

# -------------------------------------------------------------------------
# calculate Laplace spectrum
# 原始laplacian矩阵
def unnormalized_laplacian(adj_matrix):
    # 先求度矩阵
    R = np.sum(adj_matrix, axis=1)
    degreeMatrix = np.diag(R)
    return degreeMatrix - adj_matrix


# # 对称归一化的laplacian矩阵
# def normalized_laplacian(adj_matrix):
#     R = np.sum(adj_matrix, axis=1)
#     R_sqrt = 1 / np.sqrt(np.abs(R))
#     D_sqrt = np.diag(R_sqrt)
#     I = np.eye(adj_matrix.shape[0])
#     return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

# -------------------------------------------------------------------------
# for i in range(1, 91):
#     lpls_eigenvector_list = []
#     cor = np.load(f"D:/Code/Data/PhysioNet/cor/S{i}.npy") # (360, 5, 64, 64)
# #     cor[np.isnan(cor)] = 10e-11
# #     contain_nan = (True in np.isnan(cor))
# #     print(contain_nan)
# #     contain_inf = (True in np.isinf(cor))
# #     print(contain_inf)
#     for j in range(len(cor)):
#         f_list = []
#         for c in range(5):
#             # 计算laplacian矩阵
#             nor_lp = unnormalized_laplacian(cor[j, c])
#     #         # 计算laplacian矩阵的特征值和特征向量
#             lpls_eigenvalue, lpls_eigenvector = np.linalg.eig(nor_lp)
#             lpls_eigenvalue = np.repeat(lpls_eigenvalue[:, np.newaxis], 64, axis=1)
#             lpls_eigenvector = np.stack((lpls_eigenvalue, lpls_eigenvector))    # (2, 64, 64)
#             f_list.append(lpls_eigenvector)
#         lpls_eigenvector_list.append(f_list)
#     lplss_eigenvector = np.array(lpls_eigenvector_list)
#     print(lplss_eigenvector.shape)
#     np.save(f"D:/Code/Data/PhysioNet/lpls_eigenvector/S{i}.npy", lplss_eigenvector)

# -------------------------------------------------------------------------
# for i in range(1, 91):
#     data = np.load(f"D:/Code/Data/PhysioNet/raw_data/S{i}.npy") # (57600, 64)
#     labels = np.load(f"D:/Code/Data/PhysioNet/label/S{i}.npy") # (360, 1)
#
#     vsplit_data = np.vsplit(data, labels.shape[0]) # (360, 160, 64)
#     vsplit_data = np.array(vsplit_data).transpose(2, 0, 1)
#
#     lpls_eigenvalue_list = []
#     for j in range(64):
#         cor = np.corrcoef(vsplit_data[j, :, :])
#         lpls_eigenvalue, lpls_eigenvector = np.linalg.eig(cor)
#         lpls_eigenvalue_list.append(lpls_eigenvalue)
#     lplss_eigenvalue = np.array(lpls_eigenvalue_list)
#     print(lplss_eigenvalue.shape)
#     np.save(f"D:/Code/Data/PhysioNet/lpls_eigenvalue/S{i}.npy", lplss_eigenvalue)

# -------------------------------------------------------------------------
# Calculate weighted eigenvalues
# for i in range(1, 11):
# weight = np.load(f"physionet/lpls_eigenvalue/S{i}.npy") # (360, 160)
# trans_weight = weight.reshape(weight.shape[0], weight.shape[1], 1)
#
# data = np.load(f"physionet/raw_data/S{i}.npy") # (57600, 64)
# labels = np.load(f"physionet/label/S{i}.npy") # (360, 1)
# vsplit_data = np.vsplit(data, labels.shape[0]) # (360, 160, 64)
#
# w_data = vsplit_data * trans_weight
# print(w_data.shape)
# np.save(f"physionet/temporal_information/S{i}.npy", w_data)

# for i in range(1, 91):
#     weight = np.load(f"D:/Code/Data/PhysioNet/lpls_eigenvalue/S{i}.npy") # (64, 360)
#     data = np.load(f"D:/Code/Data/PhysioNet/DE/S{i}.npy") # (360, 64, 5)
#     weight = normalization(weight).transpose(1, 0).reshape(weight.shape[1], weight.shape[0], 1)
#     w_data = weight * data
#     print(w_data.shape)
#     np.save(f"D:/Code/Data/PhysioNet/temporal_information/S{i}.npy", w_data)

# -------------------------------------------------------------------------
# Combine subject
w_eigenvalues_path = "D:/Code/Data/PhysioNet/temporal_information/"
lpls_eigenvector_path = "D:/Code/Data/PhysioNet/lpls_eigenvector/"
labels_path = "D:/Code/Data/PhysioNet/label/"
save_path = "D:/Code/Data/PhysioNet/hybrid_experiment/"

eigenvalues_list = []
eigenvector_list = []
labels_list = []

for i in range(1, 91):
    eigenvalues = np.load(w_eigenvalues_path + f"S{i}.npy")
    eigenvector = np.load(lpls_eigenvector_path + f"S{i}.npy")
    labels = np.load(labels_path + f"S{i}.npy")

    eigenvalues_list.append(eigenvalues)
    eigenvector_list.append(eigenvector)
    labels_list.append(labels)
    del eigenvalues, eigenvector, labels

np.save(save_path + "eigenvalues.npy", np.concatenate(eigenvalues_list))
np.save(save_path + "eigenvector.npy", np.concatenate(eigenvector_list))
np.save(save_path + "labels.npy", np.concatenate(labels_list))
