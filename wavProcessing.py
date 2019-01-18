
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import os
import numpy
from pydub import AudioSegment


def readAudioFile(path):
    """Reads an audio file located at specified path and returns a numpy array of audio samples

    NOTE: This entire function was ripped from pyAudioAnalysis.audioBasicIO.py
    All credits to the original author, Theodoros Giannakopoulos.
    The existing audioBasicIO.py relies on broken dependencies, so it is much more reliable to rip the only function
    we need to process WAV files

    Paramters
    ----------
    path : str
        The path to a given audio file

    Returns
    ----------
    Fs : int
        Sample rate of audio file
    x : numpy array
        Data points of the audio file"""

    extension = os.path.splitext(path)[1]

    try:
        # Commented below, as we don't need this
        # #if extension.lower() == '.wav':
        #     #[Fs, x] = wavfile.read(path)
        # if extension.lower() == '.aif' or extension.lower() == '.aiff':
        #     s = aifc.open(path, 'r')
        #     nframes = s.getnframes()
        #     strsig = s.readframes(nframes)
        #     x = numpy.fromstring(strsig, numpy.short).byteswap()
        #     Fs = s.getframerate()
        if extension.lower() == '.mp3' or extension.lower() == '.wav' or extension.lower() == '.au' or extension.lower() == '.ogg':
            try:
                audiofile = AudioSegment.from_file(path)
            except:
                print("Error: file not found or other I/O error. "
                      "(DECODING FAILED)")
                return -1 ,-1

            if audiofile.sample_width == 2:
                data = numpy.fromstring(audiofile._data, numpy.int16)
            elif audiofile.sample_width == 4:
                data = numpy.fromstring(audiofile._data, numpy.int32)
            else:
                return -1, -1
            Fs = audiofile.frame_rate
            x = []
            for chn in list(range(audiofile.channels)):
                x.append(data[chn::audiofile.channels])
            x = numpy.array(x).T
        else:
            print("Error in readAudioFile(): Unknown file type!")
            return -1, -1
    except IOError:
        print("Error: file not found or other I/O error.")
        return -1, -1

    if x.ndim == 2:
        if x.shape[1] == 2:
            x = x.flatten()

    return Fs, x


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

    (rate, data) = readAudioFile(filename)

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


def make_line_plot(data, x_label="Data", y_label="Data Point"):
    """Creates a line plot of data, where each point on the plot is (i, data[i])

    Parameters
    ----------
    data : numpy array
        Any type of homogeneous numerical data
    x_label : str
        The label to put on the independent axis
    y_label : str
        The label to put on the dependent axis

    Returns
    ----------
    None"""

    y = data
    x = range(len(y))

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y)
    plt.show()


def get_trimmed_features(words, num_recordings, base_path="", energy_threshold=0.001):
    """Calculates features for a list of words, returning trimmed data based on a frame energy threshold

    Assumes all audio recordings are in the same directory base_path, and all recordings are WAV format.
    Calculates features for every recording and returns them in a hierarchical array to be fed into a neural network.
    The number of frames for each word type is the same for all recordings of that word type, as determined by
    the energy threshold for each frame.

    Parameters
    ----------
    words : [str]
        A list of distinct words
        It is assumed that audio files will have path base_path/(word)(num).wav
        Where word is one of the words in the words parameter
    num_recordings : [int]
        A list of integers >= 1
        List must have same length as words
        For word words[i], there should be num_recordings[i] distinct recordings/files of that word
        It is assumed that audio files will have path base_path/(word)(num).wav
        Where num is in the range of 1 to num_recordings
    base_path : str
        The base path that will be appended to all audio file paths as a prefix
        This is where the directory of audio files would be specified
    energy_threshold : float
        Minimum energy for a given frame to be considered relevant
        i.e. if a frame is loud enough or contains enough information to impact the data set

    Returns
    ----------
    features_by_word : numpy array
        Cell array of same length as words
        Ordering of cells is determined by the order of the words in words parameter
        The ith cell has num_recordings[i] elements
        Each element in a cell is an array of equal lengths, with each element in said array containing all relevant
        frames
        Within each frame are the 34 features extracted by pyAudioAnalysis"""

    features_by_word = []
    for i in range(len(words)):
        indexes = []
        feature_array = []
        for j in range(1, num_recordings[i] + 1):
            # Determine the path
            path = base_path + words[i] + str(j) + ".wav"
            (rate, data) = get_sig(path)
            # features is all the audio features for a given file
            features = get_st_features(data, rate)[0]
            # features[1] is total frame energies
            # energy threshold of 0.001 is arbitrary
            indexes.append(relevant_indexes(features[1], energy_threshold))
            # Add features for this specific audio file to the feature array for this word
            feature_array.append(features)
        # Finds the minimum index of all start indexes
        min_index = sorted(indexes, key=lambda x: x[0])[0][0]
        # Finds the max index of all end indexes
        max_index = sorted(indexes, key=lambda x: x[1])[::-1][0][1]
        # Debug print statements commented out
        # print("min, max index for word", words[i])
        # print(min_index, max_index)
        # Only take the frames between min index and max index for each sample word
        # Note: Potential for a bug; if maxIndex is outside the length of its frame array
        # To fix, need to pad the shorter recordings with extra data
        features_by_word.append([x[0:34, min_index:max_index].transpose() for x in feature_array])
        # print(numpy.shape(features_by_word[i]))
    # features_by_word is an array of len(words) cells
    # Each cell has num_recordings[i] elements corresponding to the number of recordings of each word words[i]
    # Each recording has the same number of frames for a given word, as determined by minIndex and maxIndex
    # for a given word.
    # Finally, each frame contains the 34 features from that frame's raw data samples
    return features_by_word


word_list = ["light", "off", "on", "slack", "tv"]
# Could change this to numbers between 1 and 30 to see how it handles more or less data
nums = [30, 30, 30, 30, 30]
# The base_directory might be different for windows users
base_directory = "ModernOTData/"
output = get_trimmed_features(word_list, nums, base_directory)

# energy_values is a sequential list of all energy values over all recordings
energy_values = []

# Should print 5
print("There are", len(output), "different words")
for word_num in range(len(output)):
    # Should print 30
    print("There are", len(output[word_num]), "different recordings for word", word_list[word_num])
    for recording_num in range(len(output[word_num])):
        # Print number of frames for each recording
        # Should be equal for all words
        print("# frames:", len(output[word_num][recording_num]), "in recording #", str(recording_num+1), "for word",
              word_list[word_num])
        for frame in output[word_num][recording_num]:
            # Should be 34 features for each frame
            # print(len(frame))
            # frame[1] is the energy for that frame
            energy_values.append(frame[1])

# Sample plot of energies across every recording
make_line_plot(energy_values, "Frame Number", "Energy")
