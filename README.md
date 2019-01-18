# Modern-OT-Warmup
Warm-up project for the Modern OT project. Couple of nerds classifying words :sound: :mega:

To get started:
```python
pip install pyAudioAnalysis matplotlib numpy
```

The meat & potatoes of this code is in `segmentation.py`, but first, you'll need to make some changes to the 
pyAudioAnalysis library.

## Modifying pyAudioAnalysis

As only mono audio is required, a revision is made to `audioBasicIO.py`, line 110.
This original code is:
```python
x = []
for chn in list(range(audiofile.channels)):
    x.append(data[chn::audiofile.channels])
x = numpy.array(x).T
```         
Which becomes:
```python
x = numpy.array(data[0::audiofile.channels]).T
```


When performing midterm segmentation, one will run into errors when the midterm window size and step are too little.
This results because the original author of pyAudioAnalysis has hardcoded short term windows and steps of 50ms
 for HMM training and 100ms for classification as
part of the short term feature extraction. To get around this, all mentions of the `mtFeatureExtraction` function in 
`audioFeatureExtraction.py` should be revised for increased flexibility.

All function headers in `audioSegmentation.py` that call this function now include optional st_win and st_step parameters, 
which are then used in the function call to `mtFeatureExtraction`.

There's an annoying print statement in `audioSegmentation.py`, line 488. Recommend commenting it out.

## Using the Project
This project uses Hidden Markov Models (HMMs) to segment audio clips based on arbitrary categorizations. This is useful
for not only classifying words, but also in identifying any characteristic in audio clips that can be labelled. An
example of this is differentiating between speech and music. For our project, we will use the library to differentiate
between regular and irregular patterns in speech. More specifically, we will remove portions of audio that have
irregular cadence, then patching together the remaining valid audio so it can be more easily processed by traditional
voice-to-text applications, such as Google Assistant.

To accomplish this, an HMM is trained and then tested using annotated audio files. Audio files must be `.wav` format.
Annotation files are plaintext `.segments` files that follow a comma separated value format as follows
```text
TIME_START1,TIME_END1,LABEL1
TIME_START2,TIME_END2,LABEL2
TIME_START3,TIME_END3,LABEL3
...
```
The `.segments` file must share the same name as the `.wav` file. Label names are arbitrary, so long as they are
consistent amongst portions of audio which share characteristics corresponding to the label.

Place all training files and annotation in `/TrainHMM/`, and all testing files and annotation in `/TestHMM/`.
