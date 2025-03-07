import math
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter

'''
file name: DE_3D_Feature
input: the path of saw EEG file in SEED-VIG dataset
output: the 3D feature of all subjects
'''


# step1: input raw data
# step2: decompose frequency bands
# step3: calculate DE
# step4: stack them into 3D featrue


def butter_bandpass(low_freq, high_freq, fs, order=5):
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# calculate DE
def calculate_DE(saw_EEG_signal):
    variance = np.var(saw_EEG_signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


# filter for 5 frequency bands
def butter_bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_bandpass(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


def decompose_to_DE(file):
    # read data  sample * channel [1416000, 17]
    data = np.load(file)
    # sampling rate
    frequency = 160
    # samples 1416000
    samples = data.shape[0]
    # 100 samples = 1 DE
    num_sample = int(samples / 160)
    channels = data.shape[1]
    bands = 5
    # init DE [141600, 17, 5]
    DE_3D_feature = np.empty([num_sample, channels, bands])

    temp_de = np.empty([0, num_sample])

    for channel in range(channels):
        trial_signal = data[:, channel]
        # get 5 frequency bands
        delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
        theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
        alpha = butter_bandpass_filter(trial_signal, 8, 12, frequency, order=3)
        beta = butter_bandpass_filter(trial_signal, 12, 30, frequency, order=3)
        gamma = butter_bandpass_filter(trial_signal, 30, 50, frequency, order=3)
        # DE
        DE_delta = np.zeros(shape=[0], dtype=float)
        DE_theta = np.zeros(shape=[0], dtype=float)
        DE_alpha = np.zeros(shape=[0], dtype=float)
        DE_beta = np.zeros(shape=[0], dtype=float)
        DE_gamma = np.zeros(shape=[0], dtype=float)
        # DE of delta, theta, alpha, beta and gamma
        for index in range(num_sample):
            DE_delta = np.append(DE_delta, calculate_DE(delta[index * 160:(index + 1) * 160]))
            DE_theta = np.append(DE_theta, calculate_DE(theta[index * 160:(index + 1) * 160]))
            DE_alpha = np.append(DE_alpha, calculate_DE(alpha[index * 160:(index + 1) * 160]))
            DE_beta = np.append(DE_beta, calculate_DE(beta[index * 160:(index + 1) * 160]))
            DE_gamma = np.append(DE_gamma, calculate_DE(gamma[index * 160:(index + 1) * 160]))
        temp_de = np.vstack([temp_de, DE_delta])
        temp_de = np.vstack([temp_de, DE_theta])
        temp_de = np.vstack([temp_de, DE_alpha])
        temp_de = np.vstack([temp_de, DE_beta])
        temp_de = np.vstack([temp_de, DE_gamma])

    temp_trial_de = temp_de.reshape(-1, 5, num_sample)
    temp_trial_de = temp_trial_de.transpose([2, 0, 1])
    DE_3D_feature = np.vstack([temp_trial_de])

    return DE_3D_feature


if __name__ == '__main__':
    # Fill in your SEED-VIG dataset path
    filePath = 'raw_data/'
    dataName = ['S1.npy', 'S2.npy', 'S3.npy', 'S4.npy', 'S5.npy', 'S6.npy', 'S7.npy', 'S8.npy', 'S9.npy', 'S10.npy',
                'S11.npy', 'S12.npy', 'S13.npy', 'S14.npy', 'S15.npy',
                'S16.npy', 'S17.npy', 'S18.npy', 'S19.npy', 'S20.npy', 'S21.npy', 'S22.npy', 'S23.npy', 'S24.npy',
                'S25.npy', 'S26.npy', 'S27.npy', 'S28.npy', 'S29.npy', 'S30.npy',
                'S31.npy', 'S32.npy', 'S33.npy', 'S34.npy', 'S35.npy', 'S36.npy', 'S37.npy', 'S38.npy', 'S39.npy',
                'S40.npy', 'S41.npy', 'S42.npy', 'S43.npy', 'S44.npy', 'S45.npy',
                'S46.npy', 'S47.npy', 'S48.npy', 'S49.npy', 'S50.npy', 'S51.npy', 'S52.npy', 'S53.npy', 'S54.npy',
                'S55.npy', 'S56.npy', 'S57.npy', 'S58.npy', 'S59.npy', 'S60.npy',
                'S61.npy', 'S62.npy', 'S63.npy', 'S64.npy', 'S65.npy', 'S66.npy', 'S67.npy', 'S68.npy', 'S69.npy',
                'S70.npy', 'S71.npy', 'S72.npy', 'S73.npy', 'S74.npy', 'S75.npy',
                'S76.npy', 'S77.npy', 'S78.npy', 'S79.npy', 'S80.npy', 'S81.npy', 'S82.npy', 'S83.npy', 'S84.npy',
                'S85.npy', 'S86.npy', 'S87.npy', 'S88.npy', 'S89.npy', 'S90.npy']

    X = np.empty([0, 64, 5])

    for i in range(len(dataName)):
        dataFile = filePath + dataName[i]
        print('processing {}'.format(dataName[i]))
        # every subject DE feature
        DE_feature = decompose_to_DE(dataFile)
        np.save("D:/Code/Data/PhysioNet/DE/S{}.npy".format(i + 1), DE_feature)
        # all subjects
        X = np.vstack([X, DE_feature])

    # save .npy file
    # np.save("E:/博士成果/跟吴老师的第一篇文章/BCI-III-IIIa/DE/All_Subject.npy", X)

