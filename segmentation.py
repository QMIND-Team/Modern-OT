from lib import audioAnalysis as aa
from scipy.io import wavfile as wf
import numpy as np
import wavProcessing as wp
import pickle as cPickle
import glob
import os

# Directory where the annotated training data is located
trainDir = 'ModernOTData/TrainHMM'
# Name to store the final HMM under
hmmName = 'TrainedHMM'
# Location of the test files, including annotations and .wav
testDir = 'ModernOTData/TestHMM'


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
# overfitting.
# Also, the parameters used to train the HMM might not be the best ones to test/use the HMM on, but I think this is
# unlikely.

# Another note: Our HMM data is pickled with mtWinSize and mtWinStep included. Maybe stWinSize and stWinStep should also
# be included in the saved data. This way, aS.hmmSegmentation doesn't need the extra parameters

# Once the HMM is trained, audio file editing must be addressed. As the hmmSegmentation function returns a list of
# labels for an audio file, there should be an easy way to remove samples between certain times and save an edited
# version of the audio file to the disk


def optimizeParameters(stWinSizeMin=0.02, stWinSizeMax=0.05, stWinSizeInterval=0.005, stWinStepFactor=2,
                       stWinStepNumIntervals=10, mtWinSizeMin=0.1, mtWinSizeMax=0.3, mtWinSizeInterval=0.02,
                       mtWinStepFactor=2, mtWinStepNumIntervals=10, hmmTempName="tempHMM"):
    """
    This function does everything I outlined above. Currently too lazy to actually format it.
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
                        # If this config is better than the existing,
                        if acc > bestAcc:
                            bestAcc = acc
                            bestConfig = (mtWinSize, mtWinStep, stWinSize, stWinStep)
                    except ValueError:
                        # Some configurations raise a value error. Haven't pinned down exactly which ones those are
                        # But there is a pattern. Skipping such configs could optimize the processing
                        print("Bad Config:")
                        print(mtWinSize, mtWinStep, stWinSize, stWinStep)
                    stWinStep += stWinStepInterval
                stWinSize += stWinSizeInterval
            mtWinStep += mtWinStepInterval
        mtWinSize += mtWinSizeInterval

    # Delete the temp HMM
    os.remove(hmmTempName)

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
        wav_file = f
        gt_file = f.replace('.wav', '.segments')
        if not os.path.isfile(gt_file):
            continue
        _, _, acc, _ = aa.hmmSegmentation(wav_file, hmmTestName, False, gt_file)
        totalAcc += acc
        totalFiles += 1
    return totalAcc/totalFiles


def segmentAudioFile(path, trainedHMMName, flag):
    """
    Edits an audio file using a trained HMM to segment audio files and remove the specified flags from the audio
    :param path: String
        Path to audio file being segmented, including the .wav extension
    :param trainedHMMName: String
        Name of the file where the pickled HMM is stored
    :param flag: String
        Label to be removed from audio, corresponding to a label in the annotated audio files the HMM was trained on
    :return: (int, list)
        Return a tuple containing the signal sample rate and the modified signal
    """
    # Read the HMM for window size/step info
    try:
        fo = open(trainedHMMName, "rb")
        _ = cPickle.load(fo)
        _ = cPickle.load(fo)
        _ = cPickle.load(fo)
        mtWinStep = cPickle.load(fo)
    except IOError:
        print("Couldn't open HMM")
        return -1, -1
    fo.close()

    # Read the audio file
    try:
        sampleRate, sig = wp.readAudioFile(path)
    except IOError:
        print("Couldn't read audio file")
        return -1, -1

    x = np.empty((0, 1), dtype=sig.dtype)
    # Segment the audio file
    flags, classes, _, _ = aa.hmmSegmentation(path, trainedHMMName)
    # Convert the supplied flag to a numerical index
    flagInd = classes.index(flag)
    # TODO: Ensure this algorithm works in the case where there is overlap (mtWinSize!=mtWinStep)
    # This could become an issue when there is overlap and we near the end of a file. Flags towards the end may not be
    # analyzed correctly
    samplesPerFlag = round(mtWinStep * sampleRate)
    for i, f in enumerate(flags):
        # Case 1: found audio we want to keep, and not at end of signal
        if f != flagInd and i < len(flags) - 1:
            nextData = sig[i*samplesPerFlag:i*samplesPerFlag + samplesPerFlag]
            x = np.append(x, np.array(nextData).reshape(len(nextData), 1))
        # Case 2: Found audio we ant to keep, but at end of file
        elif f != flagInd:
            nextData = sig[i*samplesPerFlag:]
            x = np.append(x, np.array(nextData).reshape(len(nextData), 1))
    return sampleRate, x


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


# If the an HMM has not already been trained
if not os.path.isfile(hmmName):
    # Get the best config for short and mid term window size and step
    # Current parameters chosen solely for speed, probably not the best for actual testing
    config = optimizeParameters(stWinSizeMin=0.02, stWinSizeMax=0.025, stWinSizeInterval=0.005, stWinStepFactor=2,
                                stWinStepNumIntervals=2, mtWinSizeMin=0.1, mtWinSizeMax=0.2, mtWinSizeInterval=0.05,
                                mtWinStepFactor=2, mtWinStepNumIntervals=2, hmmTempName="tempHMM")
    # Create a trained HMM with the best configuration.
    aa.trainHMM_fromDir(trainDir, hmmName, config[0], config[1], config[2], config[3])

# Get an edited audio signal. The third parameter is whatever label should be removed from the audio clip
rate, signal = segmentAudioFile("ModernOTData/test.wav", hmmName, "light")
# Save the audio signal
writeAudioFile("output.wav", rate, signal)
