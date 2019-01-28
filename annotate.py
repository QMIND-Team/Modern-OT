"""
Generates stubs for annotation files by detecting where a clip is silent vs. loud
"""
from lib import audioAnalysis as aa
from lib import audioFeatureExtraction as aF
import os


path = "ModernOTData/KeywordTest/KeywordTest2.wav"

outputPath = path.split(".")[0] + ".segments"
output = "0.00,"
nextOutput = ""

rate, data = aa.readAudioFile(path)
thresholdSilence = 800
thresholdWord = 300
step = round(rate*0.01)
lastIdx = 0.00
isSilence = True
for i in range(int(len(data)/step)):
    # Find energy in the next steps
    if not isSilence:
        if (i + 10) * step > len(data):
            e = abs(aF.stEnergy(data[i * step:]))
        else:
            # Calculate average energy over next 10 frames
            eTemp=0
            for j in range(i,i+10):
                eTemp += abs(aF.stEnergy(data[j*step:(j + 1) * step]))
            e = eTemp/10
    else:
        e = abs(aF.stEnergy(data[i * step:(i + 2) * step]))
    if isSilence and e > thresholdSilence:
        idx = (i * step) / rate
        isSilence = False
        nextOutput = str(idx) + ",silence\n" + str(idx) + ","
        lastIdx = idx
    elif (not isSilence) and e < thresholdWord:
        idx = (i * step) / rate
        if idx - lastIdx < 0.1:
            isSilence = True
            continue
        output += nextOutput + str(idx) + ",\n" + str(idx) + ","
        isSilence = True
        lastIdx = idx

# Add final silence
output += str(round(len(data)/rate)) + ",silence"

# Write output
f = open(outputPath, "w")
f.write(output)
f.close()
