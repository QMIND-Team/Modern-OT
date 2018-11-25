from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy


def get_sig(filename):
    """Gets a signal from an audio file

    Parameters
    ----------
    filename : string
        Name of the WAV audio file, including extension

    Returns
    ----------
    rate : int
        Sample rate of the audio file
    signal : numpy array
        NumPy array containing all sample points in the audio file.
        The NumPy dtype in the array depends on the format of the WAV file."""

# NOTE: pyAudioAnalysis.audioBasicIO.readAudioFile has a bug where it returns an array of dimensions (n, 2)
# Fix: Flatten it
    (rate, data) = audioBasicIO.readAudioFile(filename)

    return rate, data


def get_st_features(signal, rate, window_step=0.025, window_length=0.05):
    """Computes all 34 features for each window in a given signal

    Parameters
    ----------
    signal : numpy array
        All sample points for the audio signal
        Can be any type of number
    rate : int
        Sample rate of the audio signal, in Hz
    window_step : float
        Time step between each successive window, in seconds
        Default: 0.025 (25 ms)
    window_length : float
        Length of each window, in seconds
        Should generally be greater than windowStep to allow for overlap between frames
        Default: 0.05 (50 ms)

    Returns
    ----------
    features : numpy array
        NumPy array of size (number of windows) * 34
        Each row in mfcc_features contains all the features for a single frame
    feature_names : [str]
        Names of each feature located at specified index"""

    sample_step = int(rate*window_step)
    sample_length = int(rate*window_length)

    (features, feature_names) = audioFeatureExtraction.stFeatureExtraction(signal, rate, sample_length, sample_step)

    return features, feature_names


def relevant_indexes(data, min_threshold):
    """Finds first and last index where data > min_threshold

    To find the start and end indexes of the frames where there is some noise
    Could be useful to take many audio clips and find the lowest start index and highest end index common between
    all audio clips. This would be useful if the ML code must take a fixed # of input layer data points

    Parameters
    ----------
    data : numpy array
        Energy levels of multiple frames
    min_threshold : float
        Minimum threshold value that each data is compared to

    Returns
    ----------
    start_index : int
        First index in data with a value greater than min_threshold
    end_index : int
        Last index in data with a value greater than min_threshold"""

    start_index = 1
    end_index = len(data) - 1

    for i in range(len(data)):
        if data[i] > min_threshold:
            start_index = i
            break

    for i in range(len(data)):
        if data[::-1][i] > min_threshold:
            end_index = i
            break

    return start_index, end_index


def make_line_plot(data):

    #(startIdx, endIdx) = relevant_indexes(data, min_threshold)
    #print(data[0])
    y = data
    x = range(len(y))

    plt.xlabel("Data point")
    plt.ylabel("Data")
    plt.plot(x, y)
    plt.show()

