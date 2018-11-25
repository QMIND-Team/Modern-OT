import wavProcessing
import h5py

#import and tag data
for i in range(0:30)
    (rate, x_data[i]) = get_sig("ModernOTData/tv" + i + ".wav")
    data[i] = relevant_indexes(data[i], )
    (features, f_names) = get_st_features(x_data[i], rate)
    data[i] = (features, 0)
for i in range(30:60)
    (rate, x_data[i]) = get_sig("ModernOTData/on" + i + ".wav")
    data[i] = relevant_indexes(data[i], )
    (features, f_names) = get_st_features(x_data[i], rate)
    data[i] = (features, 1)
for i in range(60:90)
    (rate, x_data[i]) = get_sig("ModernOTData/off" + i + ".wav")
    data[i] = relevant_indexes(data[i], )
    (features, f_names) = get_st_features(x_data[i], rate)
    data[i] = (features, 2)
for i in range(90:120)
    (rate, x_data[i]) = get_sig("ModernOTData/slack" + i + ".wav")
    data[i] = relevant_indexes(data[i], )
    (features, f_names) = get_st_features(x_data[i], rate)
    data[i] = (features, 3)
for i in range(120:150)
    (rate, x_data[i]) = get_sig("ModernOTData/light" + i + ".wav")
    data[i] = relevant_indexes(data[i], )
    (features, f_names) = get_st_features(x_data[i], rate)
    data[i] = (features, 4)

#shuffle and split data
np.random.shuffle(data)
train = data[:int(len(data)*0.8)]
test = data[int(len(data)*0.8):]
trainX = train[:-1]
trainY = train[-1:]
testX = test[:-1]
testY = [-1:]

#save data as an hdf5 file
h5f = h5py.file('data.h5', 'w')
h5f.create_dataset('testx', data=testX)
h5f.create_dataset('testy', data=testY)
h5f.create_dataset('trainx', data=trainX)
h5f.create_dataset('trainy', data=trainY)

