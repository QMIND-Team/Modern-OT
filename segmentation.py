from pyAudioAnalysis import audioSegmentation as aS

# Directory where the annotated training data is located
trainDir = 'ModernOTData/RawAnnotated'
# Name to store the HMM under
hmmName = 'hmmTemp1'
# Location of the test file
testFile = 'ModernOTData/TrainHMM/light-off.wav'
# Location of the .segments for the test file
testFileSegments = 'ModernOTData/TrainHMM/light-off.segments'

aS.trainHMM_fromDir(trainDir, hmmName, 0.1, 0.1)
aS.hmmSegmentation(testFile, hmmName, True, testFileSegments)
