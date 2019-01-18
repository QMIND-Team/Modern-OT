# Modern-OT-Warmup
Warm-up project for the Modern OT project. Couple of nerds classifying words :sound: :mega:

To get started:
```python
pip install pyAudioAnalysis matplotlib numpy
```

There is a bug in pyAudioAnalysis/audioBasicIO.py, line 110:
```python
if x.shape[1]==1:
```
This line needs to be changed to
```python
if x.shape[1]==2:
```