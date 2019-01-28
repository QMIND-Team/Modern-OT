import sounddevice as sd
import numpy as np
import segmentation as seg
from lib import audioFeatureExtraction as aF
import http.server
import socketserver

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


def recordCheck(block, overlap, threshold, hmmData, sampleRate, keywordLabel):
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
    :param hmmData: dict
        Dict containing all of the HMM data
    :param sampleRate: int
        Sample rate of all the audio
    :param keywordLabel: String
        Keyword to look for in audio
    :return: None
    """
    commandsFound = 0
    # Keep track of whether we're recording a command
    isCommand = False
    lastBlock = sd.rec(int(sampleRate * block))
    sd.wait()
    audio = lastBlock
    # Record loop
    while True:
        # Record next block
        nextBlock = sd.rec(int(sampleRate * block))

        # If we are recording a command and the keyword is found
        if isCommand and seg.containsLabel(audio, sampleRate, hmmData, keywordLabel):
            print("Command completed")
            # Remove keyword at end of signal
            audio = seg.trimKeyword(audio, sampleRate, hmmData, keywordLabel, True)
            # Save the command
            seg.writeAudioFile("command" + str(commandsFound) + ".wav", sampleRate, audio)
            isCommand = False
            overlapNextBlock = False
            commandsFound += 1
        # If there is an overlap, continue recording
        elif checkForSound(lastBlock, overlap, threshold):
            print("Continuing a block")
            overlapNextBlock = True
        # Otherwise, check if a keyword as said
        else:
            # This is where we would check for a keyword, and act accordingly
            # TODO: If a keyword were to appear twice in a single block, this would only identify one occurrence
            if seg.containsLabel(audio, sampleRate, hmmData, keywordLabel):
                print("Command initiated")
                isCommand = True
                overlapNextBlock = True
                # Only keep the audio up to where the keyword starts
                audio = seg.trimKeyword(audio, sampleRate, hmmData, keywordLabel)
            else:
                print("Ended a block")
                overlapNextBlock = False

        # Wait until next block is done recording
        sd.wait()
        # Only want to start a new recording if there's no overlap, and there is no command
        if not overlapNextBlock:
            audio = np.empty((0, 1), dtype=np.int16)
        audio = np.append(audio, nextBlock.reshape((len(nextBlock), 1)))
        lastBlock = nextBlock


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


def start(hmmPath, keywordLabel):
    """
    Main function. Supply an HMM trained to recognize the given keyword label, and watch the magic happen
    :param hmmPath: String
        Path + name of the trained HMM
    :param keywordLabel: String
        Label the HMM is trained to recognize
    :return: None
    """
    # Read HMM
    hmmData = seg.readHMM(hmmPath)

    # Set constants
    # Length of time to record each block for, in seconds
    blockTime = 1
    # last % of the recording to check for any audio that will overlap between blocks
    overlapCheckPortion = 0.1
    # Sample rate of audio to collect
    fs = 44100
    # The energy threshold that is considered significant
    energyThreshold = 2500

    # Set defaults for sounddevice library
    sd.default.dtype = np.int16
    sd.default.samplerate = fs
    sd.default.channels = 1

    # Call main runtime loop
    recordCheck(blockTime, overlapCheckPortion, energyThreshold, hmmData, fs, keywordLabel)


def createLocalHost(port=8888):
    """
    Creates a local host http server for project at address "http://localhost:port" where port is the port number. One
    can access sub directories of project directory of local host by adding that to the end of the address. For example
    to access .wav file in data folder one would go to address "http://localhost:port/DataFolder/data.wav"
    :param port: the port number of the local host
    :return: void
    """
    Handler = http.server.SimpleHTTPRequestHandler

    httpd = socketserver.TCPServer(("", port), Handler)
    print("Currently Serving Port:", port)
    httpd.serve_forever()
