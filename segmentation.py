from lib import audioAnalysis as aa
from scipy.io import wavfile as wf
import numpy as np
import pickle as cPickle
import glob
import os

# If we were to iterate through these parameters with different values for every short- and mid-term window step
# and size, we would be able to optimize the HMM training for maximum accuracy
# Depending on the training and test data, and what we're looking to optimize and segment for, we will be able to
# Get a high accuracy in segmenting clips to remove irregular speech patterns depending on the speakers training data

# Current proposal:
# Short term window size could range between 20ms and 100ms, in intervals of 2-5ms
# Call these stWinSizeMin and stWinSizeMax and stWinSizeInterval. The 20-100ms is based on research, the interval is not

# Short term window step could vary such that there is no overlap (stWinSize) down to maybe 75% overlap, or more
# So Short term window step could vary between stWinSize/4 and stWinSize, possibly with 10-50 intervals between
# (Admittedly, the # of intervals I have no research to support what may be optimal, but empirical data can aid here)
# If we were to use, say 20 intervals, the interval would be (stWinSizeMax - stWinSizeMin)/20
# To generalize: stWinStepMin = stWinSize/stWinStepFactor, stWinStepMax = stWinSize,
# stWinStepInterval = (stWinStepMax - stWinStepMin)/stWinStepNumIntervals

# All of the above could be repeated for the midterm window variables
# Then, iterate through parameters to find the highest accuracy
# Also, once each set of parameters has been used to train the HMM, it should be tested on multiple files to avoid
# over-fitting.
# Also, the parameters used to train the HMM might not be the best ones to test/use the HMM on, but I think this is
# unlikely.

# Another note: Our HMM data is pickled with mtWinSize and mtWinStep included. Maybe stWinSize and stWinStep should also
# be included in the saved data. This way, aS.hmmSegmentation doesn't need the extra parameters

# Once the HMM is trained, audio file editing must be addressed. As the hmmSegmentation function returns a list of
# labels for an audio file, there should be an easy way to remove samples between certain times and save an edited
# version of the audio file to the disk


def optimizeParameters(hmmTempName, trainDir, testDir, stWinSizeMin=0.02, stWinSizeMax=0.05, stWinSizeInterval=0.005,
                       stWinStepFactor=2, stWinStepNumIntervals=10, mtWinSizeMin=0.1, mtWinSizeMax=0.3,
                       mtWinSizeInterval=0.02, mtWinStepFactor=2, mtWinStepNumIntervals=10):
    """
    This function does everything I outlined above. Currently too lazy to actually format it.
    :param hmmTempName:
    :param trainDir:
    :param testDir:
    :param stWinSizeMin:
    :param stWinSizeMax:
    :param stWinSizeInterval:
    :param stWinStepFactor:
    :param stWinStepNumIntervals:
    :param mtWinSizeMin:
    :param mtWinSizeMax:
    :param mtWinSizeInterval:
    :param mtWinStepFactor:
    :param mtWinStepNumIntervals:
    :return:
    """
    # Constant to avoid weird floating point errors so that no configurations are skipped
    floatError = 0.00000002
    # Set defaults
    mtWinSize = mtWinSizeMin
    log = "Accuracy,mtWinSize,mtWinStep,stWinSize,stWinStep\n"
    # Keep track of best accuracy
    bestAcc = 0
    bestConfig = []
    print("Optimizing parameters. This will take a long time.")
    # Loop for mtWinSize
    while mtWinSize <= mtWinSizeMax + floatError:
        # Initialize mtWinStep at mtWinSize (no overlap)
        mtWinStep = mtWinSize/mtWinStepFactor
        mtWinStepInterval = (mtWinSize - mtWinStep)/mtWinStepNumIntervals
        # Loop for mtWinStep
        while mtWinStep <= mtWinSize + floatError:
            stWinSize = stWinSizeMin
            # Loop for stWinSize
            while stWinSize <= stWinSizeMax + floatError:
                stWinStep = stWinSize / stWinStepFactor
                stWinStepInterval = (stWinSize - stWinStep) / stWinStepNumIntervals
                # Loop for stWinStep
                while stWinStep <= stWinSize + floatError:
                    try:
                        # Parameters selected, now train HMM
                        aa.trainHMM_fromDir(trainDir, hmmTempName, mtWinSize, mtWinStep, stWinSize, stWinStep)
                        # Test HMM
                        acc = testFromDir(testDir, hmmTempName)
                        log += ",".join(["{0:.4f}".format(acc), str(round(mtWinSize,4)), str(round(mtWinStep,4)),
                                         str(round(stWinSize,4)), str(round(stWinStep, 4))]) + "\n"
                        print("Last accuracy: {0:.2f}%".format(acc*100))
                        # If this config is better than the existing,
                        if acc > bestAcc:
                            bestAcc = acc
                            bestConfig = (mtWinSize, mtWinStep, stWinSize, stWinStep)
                    except ValueError:
                        # Some configurations raise a value error. Haven't pinned down exactly which ones those are
                        # But there is a pattern. Skipping such configs could optimize the processing
                        print("Bad Config: ValueError")
                        print(mtWinSize, mtWinStep, stWinSize, stWinStep)
                    except IndexError:
                        # This is a weird one, not sure what causes it
                        print("Bad Config: IndexError")
                        print(mtWinSize, mtWinStep, stWinSize, stWinStep)
                    stWinStep += stWinStepInterval
                stWinSize += stWinSizeInterval
            mtWinStep += mtWinStepInterval
        mtWinSize += mtWinSizeInterval

    # Delete the temp HMM
    os.remove(hmmTempName)

    f = open("log.txt", "w")
    f.write(log)
    f.close()

    print("Best accuracy is {0:.2f}%".format(bestAcc*100))
    print("Best config is:")
    print("mtWinSize =", bestConfig[0])
    print("mtWinStep =", bestConfig[1])
    print("stWinSize =", bestConfig[2])
    print("stWinStep =", bestConfig[3])
    return bestConfig


def testFromDir(path, hmmTestName, numFiles=-1):
    """
    Calculates the average segmentation accuracy of multiple test files in a directory.
    Requires that all .wav files in the directory are annotated with an accompanying .segments file.
    :param path: String
        Path to the directory
    :param hmmTestName: String
        The name of the trained HMM to be tested
    :param numFiles: int
        (optional) The number of the files in the directory to test with. If not specified, all files will be tested.
        If not all files are being used for testing, they will be randomly selected
    :return: float
        Average segmentation accuracy
    """
    if numFiles == -1:
        allFiles = True
    else:
        allFiles = False

    totalAcc = 0
    totalFiles = 0
    for i, f in enumerate(glob.glob(path + os.sep + '*.wav')):
        if not allFiles and i >= numFiles:
            break
        # for each WAV file
        fs, x = aa.readAudioFile(f)
        hmm = readHMM(hmmTestName)
        gt_file = f.replace('.wav', '.segments')
        if not os.path.isfile(gt_file):
            continue
        _, _, acc, _ = aa.hmmSegmentation(x, fs, hmm, False, gt_file)
        totalAcc += acc
        totalFiles += 1
    return totalAcc/totalFiles


def segmentAudioSignal(signal, sampleRate, hmmData):
    """
    Segments an audio clip using a trained HMM. This function handles using the trained HMM.
    Does not read, write, nor edit audio files
    :param signal: ndarray
        Audio signal
    :param sampleRate: int
        Sample rate of signal
    :param hmmData: dict
        Dict containing all of the HMM data
    :return: (int, list)
        Return a tuple containing the signal sample rate and the modified signal
    """
    # Segment the audio file
    flags, classes, _, _ = aa.hmmSegmentation(signal, sampleRate, hmmData)
    return flags, classes


def removeSegments(signal, sampleRate, flags, classes, winStep, labelsToRemove):
    """
    Creates a new signal by removing all instances of specified labels from an original signal, after original signal
    has been run through a trained HMM
    :param signal: ndarray
        The original audio signal
    :param sampleRate: int
        Sample rate of audio signal
    :param flags: [int]
        Index flag output from trained HMM
    :param classes: [String]
        Class list output from trained HMM
    :param winStep: float
        Time (seconds) corresponding to the length of one flag
    :param labelsToRemove: [String]
        Labels to remove from original signal
    :return: ndarray
        Audio signal with specified labels removed
    """
    # Create empty array for new signal
    x = np.empty((0, 1), dtype=signal.dtype)
    # Convert the supplied labels to a numerical indexes
    flagIndexes = []
    for label in labelsToRemove:
        try:
            flagIndexes.append(classes.index(label))
        except ValueError:
            print("Error: label " + label + " not in class list. Continuing without it")
            continue

    # TODO: Ensure this algorithm works in the case where there is overlap (mtWinSize!=mtWinStep)
    # This could become an issue when there is overlap and we near the end of a file. Flags towards the end may not be
    # analyzed correctly
    samplesPerFlag = round(winStep * sampleRate)
    for i, f in enumerate(flags):
        # Case 1: found audio we want to keep, and not at end of signal
        if f not in flagIndexes and i < len(flags) - 1:
            nextData = signal[i*samplesPerFlag:i*samplesPerFlag + samplesPerFlag]
            x = np.append(x, np.array(nextData).reshape(len(nextData), 1))
        # Case 2: Found audio we ant to keep, but at end of file
        elif f not in flagIndexes:
            nextData = signal[i*samplesPerFlag:]
            x = np.append(x, np.array(nextData).reshape(len(nextData), 1))
    return x


def trimKeyword(signal, sampleRate, hmmData, keywordLabel, reverse=False):
    """
    Removes audio from a signal up to and including the keyword. Works when the keyword is known to be in the signal
    :param signal: ndarray
        Audio signal known to have a keyword in it
    :param sampleRate: int
        Sample rate of audio signal
    :param hmmData: dict
        Dict containing all of the HMM data
    :param keywordLabel: String
        Keyword label known to be in signal
    :param reverse: bool
        Indicates whether to start trimming before the keyword, or after. If True, audio before the keyword is kept
    :return: ndarray
        The edited audio signal with all audio up to and including the keyword removed
    """
    # First, figure out where keyword is within the signal
    flags, classes = segmentAudioSignal(signal, sampleRate, hmmData)
    # Then, trim audio after the start of the keyword
    if reverse:
        return signal[:flags.index(classes.index(keywordLabel)) * int(sampleRate * hmmData["mtWinStep"])]
    # Or trim audio before the end of the keyword
    else:
        return signal[(len(flags) - flags[::-1].index(classes.index(keywordLabel)) + 1) *
                      int(sampleRate * hmmData["mtWinStep"]):]


def writeAudioFile(path, sampleRate, sig):
    """
    Writes the specified signal to a .wav file
    :param path: String
        Path to the file, including the .wav extension
    :param sampleRate: int
        Sample rate of the signal
    :param sig: [float]
        Signal to be written to file
    :return: None
    """
    wf.write(path, sampleRate, sig)


def trainHMM(trainDir='ModernOTData/KeywordTrain', hmmName='TrainedHMM', testDir='ModernOTData/KeywordTest'):
    """
    Train the HMM
    :param trainDir: String
        Directory where the annotated training data is located
    :param hmmName: String
        Name to store the final HMM under
    :param testDir: String
        Location of the test files, including annotations and .wav
    :return: None
    """
    # Get the best config for short and mid term window size and step
    config = optimizeParameters("tempHMM", trainDir, testDir, stWinSizeMin=0.02, stWinSizeMax=0.04,
                                stWinSizeInterval=0.01, stWinStepFactor=2, stWinStepNumIntervals=2,
                                mtWinSizeMin=0.1, mtWinSizeMax=0.3, mtWinSizeInterval=0.1, mtWinStepFactor=2,
                                mtWinStepNumIntervals=2)
    # Create a trained HMM with the best configuration.
    aa.trainHMM_fromDir(trainDir, hmmName, config[0], config[1], config[2], config[3])


def readHMM(path):
    """
    Reads an HMM at the path specified, returning a dict of the values
    :param path: String
        Path + name of pickled HMM
    :return: dict
        Dict containing all of the HMM data
    """
    # Read the HMM for window size/step info
    try:
        fo = open(path, "rb")
        hmmData = {
            "HMM": cPickle.load(fo),
            "allClasses": cPickle.load(fo),
            "mtWinSize": cPickle.load(fo),
            "mtWinStep": cPickle.load(fo),
            "stWinSize": cPickle.load(fo),
            "stWinStep": cPickle.load(fo)
        }
    except IOError:
        print("Couldn't open HMM")
        return
    fo.close()
    return hmmData


def containsLabel(signal, sampleRate, hmmData, label):
    """
    Uses a trained HMM to check if a given audio file contains a given label. Should be a lightweight function as it is
    called frequently
    :param signal: ndarray
        Audio signal
    :param sampleRate: int
        Sample rate
    :param hmmData: dict
        Dict containing HMM data
    :param label: String
        Label to check signal for
    :return: int
        Returns True if the label is detected in the audio file, False if no
    """
    flags, classes = segmentAudioSignal(signal, sampleRate, hmmData)
    try:
        index = classes.index(label)
    except ValueError:
        print("HMM not trained to recognize label" + label)
        return
    return index in flags


def quickOptimize(trainDir="ModernOTData/KeywordTrain", testDir="ModernOTData/KeywordTest", mtWinMin=0.025, mtWinMax=0.2,
                  stWinMin=0.01, stWinMax=0.025):
    bestAcc = 0
    bestConfig = []

    mtWin = mtWinMin
    # Constant to avoid weird floating point errors so that no configurations are skipped
    floatError = 0.00000002
    while mtWin < mtWinMax - floatError:
        stWin = stWinMin
        while stWin < stWinMax - floatError:
            aa.trainHMM_fromDir(trainDir, "tempHMM", mtWin, mtWin, stWin, stWin/2)
            acc = testFromDir(testDir, "tempHMM")
            print("Current accuracy: {0:.2f}%".format(acc*100))
            if acc > bestAcc:
                bestAcc = acc
                bestConfig = (mtWin, mtWin, stWin, stWin/2)
            stWin += 0.0025
        mtWin += 0.025

    os.remove("tempHMM")

    print("Best accuracy is {0:.2f}%".format(bestAcc * 100))
    print("Best config is:")
    print("mtWinSize =", bestConfig[0])
    print("mtWinStep =", bestConfig[1])
    print("stWinSize =", bestConfig[2])
    print("stWinStep =", bestConfig[3])
    return bestConfig
