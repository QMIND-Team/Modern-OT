import sounddevice as sd
import numpy as np
from lib import audioFeatureExtraction as aF

"""
This script is the main recording loop. It should continuously record data, spaced in blocks of a specific length of 
time. This length of time should be at least the time it takes to say a trigger word, plus some sort of buffer.
Try maybe 1 second, and work down from there. Smaller blocks have the advantage of increased responsiveness, but likely
more preprocessing
Recordings are stored as numpy ndarray, with dtype=int16 (subject to change)
Preprocessing needs to be done on the audio, namely if there is any audio recorded in the last 10% (subject to change)
then piece the next recording block together (they're all just numpy arrays, so this should be easy)
Once there is no audio recorded in the last 10% of the block, send the recorded audio to the HMM to look for the tigger
word. If the trigger word is found, trim everything before the trigger word and keep recording until there is a second
occurrence of the trigger word, and stop the recording there.
Take the entire command and send it to the HMM, cut out the trigger words, clean up the audio, remove silence, and then
take the resulting audio file and send it to the Google Assistant API.
"""

# Length of time to record each block for, in seconds
blockTime = 1
# last % of the recording to check for any audio that will overlap between blocks
overlapCheckPortion = 0.1
# Sample rate of audio to collect
fs = 44100
# The energy threshold that is considered significant
energyThreshold = 2000

# Set defaults
sd.default.dtype = np.int16
sd.default.samplerate = fs
sd.default.channels = 1


def recordCheck(block, overlap, threshold):
    """
    Runtime loop that continually records and checks the last portion of a recording of specified length for audio.
    Pieces blocks of audio together, then detects silence, and feeds that recording to the HMM to check for trigger
    words to initiate a command sequence
    :param block: float
        Length of each recording block, in seconds
    :param overlap: float
        Value between 0 and 1 representing which % of the end of each block should be checked for overlap
    :param threshold: float
        Energy threshold above which is considered significant volume
    :return: None
    """
    overlapNextBlock = False
    # Record loop
    while True:
        if not overlapNextBlock:
            audio = np.empty((0, 1), dtype=np.int16)
        # Record our next block
        nextBlock = sd.rec(int(fs * block))
        # Wait until next block is done recording
        sd.wait()
        audio = np.append(audio, nextBlock.reshape((len(nextBlock), 1)))
        # If there is an overlap, continue recording
        if checkForSound(nextBlock, overlap, threshold):
            print("Continuing a recording")
            overlapNextBlock = True
        else:
            print("Ended a recording")
            overlapNextBlock = False


def checkForSound(signal, overlap, threshold):
    """
    Checks if there is sound in the last overlap % of the signal
    :param signal: ndarray
        The signal to be checked
    :param overlap: float
        Number between 0 and 1 representing how much audio to check as a %
    :param threshold: float
        The energy threshold that is considered significant
    :return: bool
        True if there is sound, False if there is not
    """
    # Index where samples are included in spectrogram energy extraction
    ind = int(len(signal) * (1 - overlap))
    if aF.stEnergy(signal[ind:]) > threshold:
        return True
    else:
        return False

