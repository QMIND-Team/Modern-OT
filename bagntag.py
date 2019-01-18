# This file is no longer needed for our project, but it might be useful to keep
# TODO: Consider deleting this file

import wavProcessing
import h5py
import pyAudioAnalysis

#import and tag data
for i in range(1,31):
    (rate, x_data[i]) = get_sig("ModernOTData/tv" + i + ".wav")
    (features, f_names) = get_st_features(x_data[i], rate)
    data[i-1] = (features, 0)
for i in range(1,31):
    (rate, x_data[i]) = get_sig("ModernOTData/on" + i + ".wav")
    (features, f_names) = get_st_features(x_data[i], rate)
    data[i+29] = (features, 1)
for i in range(1,31):
    (rate, x_data[i]) = get_sig("ModernOTData/off" + i + ".wav")
    (features, f_names) = get_st_features(x_data[i], rate)
    data[i+59] = (features, 2)
for i in range(1,31):
    (rate, x_data[i]) = get_sig("ModernOTData/slack" + i + ".wav")
    (features, f_names) = get_st_features(x_data[i], rate)
    data[i+89] = (features, 3)
for i in range(1,31):
    (rate, x_data[i]) = get_sig("ModernOTData/light" + i + ".wav")
    (features, f_names) = get_st_features(x_data[i], rate)
    data[i+119] = (features, 4)

#shuffle and split data
np.random.shuffle(data)
train = data[:int(len(data)*0.8)]
test = data[int(len(data)*0.8):]
trainX = train[:, 0]
trainY = train[:, 1]
testX = test[:, 0]
testY = test[:, 1]

#save data as an hdf5 file
h5f = h5py.file('data.h5', 'w')
h5f.create_dataset('testx', data=testX)
h5f.create_dataset('testy', data=testY)
h5f.create_dataset('trainx', data=trainX)
h5f.create_dataset('trainy', data=trainY)

