# This script is going to be the library containing all functions from pyAudioAnalysis we'd need
# Including any revisions to the original source
# pyAudioAnalysis/audioFeatureExtraction.py is needed but doesn't have any issues in it, so it is copied to the project
# directory

import numpy
import hmmlearn.hmm
import pickle as cPickle
import glob
import os
import csv
import matplotlib.pyplot as plt
from lib import audioFeatureExtraction as aF
from pydub import AudioSegment


# START SEGMENTATION SECTION
def trainHMM_fromFile(wav_file, gt_file, hmm_model_name, mt_win, mt_step, st_win=0.05, st_step=0.05):
    '''
    This function trains a HMM model for segmentation-classification using a single annotated audio file
    ARGUMENTS:
     - wav_file:        the path of the audio filename
     - gt_file:         the path of the ground truth filename
                       (a csv file of the form <segment start in seconds>,<segment end in seconds>,<segment label> in each row
     - hmm_model_name:   the name of the HMM model to be stored
     - mt_win:          mid-term window size
     - mt_step:         mid-term window step
    RETURNS:
     - hmm:            an object to the resulting HMM
     - class_names:     a list of class_names

    After training, hmm, class_names, along with the mt_win and mt_step values are stored in the hmm_model_name file
    '''

    [seg_start, seg_end, seg_labs] = readSegmentGT(gt_file)
    flags, class_names = segs2flags(seg_start, seg_end, seg_labs, mt_step)
    [fs, x] = readAudioFile(wav_file)
    [F, _, _] = aF.mtFeatureExtraction(x, fs, mt_win * fs, mt_step * fs,
                                       round(fs * st_win), round(fs * st_step))
    start_prob, transmat, means, cov = trainHMM_computeStatistics(F, flags)
    hmm = hmmlearn.hmm.GaussianHMM(start_prob.shape[0], "diag")

    hmm.startprob_ = start_prob
    hmm.transmat_ = transmat
    hmm.means_ = means
    hmm.covars_ = cov

    fo = open(hmm_model_name, "wb")
    cPickle.dump(hmm, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(class_names, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(mt_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(mt_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(st_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(st_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()

    return hmm, class_names


def trainHMM_fromDir(dirPath, hmm_model_name, mt_win, mt_step, st_win=0.05, st_step=0.05):
    '''
    This function trains a HMM model for segmentation-classification using
    a where WAV files and .segment (ground-truth files) are stored
    ARGUMENTS:
     - dirPath:        the path of the data diretory
     - hmm_model_name:    the name of the HMM model to be stored
     - mt_win:        mid-term window size
     - mt_step:        mid-term window step
    RETURNS:
     - hmm:            an object to the resulting HMM
     - class_names:        a list of class_names

    After training, hmm, class_names, along with the mt_win
    and mt_step values are stored in the hmm_model_name file
    '''

    flags_all = numpy.array([])
    classes_all = []
    for i, f in enumerate(glob.glob(dirPath + os.sep + '*.wav')):
        # for each WAV file
        wav_file = f
        gt_file = f.replace('.wav', '.segments')
        if not os.path.isfile(gt_file):
            continue
        [seg_start, seg_end, seg_labs] = readSegmentGT(gt_file)
        flags, class_names = segs2flags(seg_start, seg_end, seg_labs, mt_step)
        for c in class_names:
            # update class names:
            if c not in classes_all:
                classes_all.append(c)
        [fs, x] = readAudioFile(wav_file)
        [F, _, _] = aF.mtFeatureExtraction(x, fs, mt_win * fs,
                                           mt_step * fs, round(fs * st_win),
                                           round(fs * st_step))

        lenF = F.shape[1]
        lenL = len(flags)
        min_sm = min(lenF, lenL)
        F = F[:, 0:min_sm]
        flags = flags[0:min_sm]

        flagsNew = []
        for j, fl in enumerate(flags):  # append features and labels
            flagsNew.append(classes_all.index(class_names[flags[j]]))

        flags_all = numpy.append(flags_all, numpy.array(flagsNew))

        if i == 0:
            f_all = F
        else:
            f_all = numpy.concatenate((f_all, F), axis=1)
    start_prob, transmat, means, cov = trainHMM_computeStatistics(f_all, flags_all)  # compute HMM statistics
    hmm = hmmlearn.hmm.GaussianHMM(start_prob.shape[0], "diag")  # train HMM
    hmm.startprob_ = start_prob
    hmm.transmat_ = transmat
    hmm.means_ = means
    hmm.covars_ = cov

    fo = open(hmm_model_name, "wb")  # save HMM model
    cPickle.dump(hmm, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(classes_all, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(mt_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(mt_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(st_win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(st_step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()

    return hmm, classes_all


def hmmSegmentation(wav_file_name, hmm_model_name, plot_res=False,
                    gt_file_name=""):
    [fs, x] = readAudioFile(wav_file_name)
    try:
        fo = open(hmm_model_name, "rb")
    except IOError:
        print("didn't find file")
        return

    try:
        hmm = cPickle.load(fo)
        classes_all = cPickle.load(fo)
        mt_win = cPickle.load(fo)
        mt_step = cPickle.load(fo)
        # Edited
        st_win = cPickle.load(fo)
        st_step = cPickle.load(fo)
    except:
        fo.close()
    fo.close()

    [Features, _, _] = aF.mtFeatureExtraction(x, fs, mt_win * fs, mt_step * fs,
                                              round(fs * st_win),
                                              round(fs * st_step))
    flags_ind = hmm.predict(Features.T)  # apply model
    if os.path.isfile(gt_file_name):
        [seg_start, seg_end, seg_labs] = readSegmentGT(gt_file_name)
        flags_gt, class_names_gt = segs2flags(seg_start, seg_end, seg_labs,
                                              mt_step)
        flagsGTNew = []
        for j, fl in enumerate(flags_gt):
            # "align" labels with GT
            if class_names_gt[flags_gt[j]] in classes_all:
                flagsGTNew.append(classes_all.index(class_names_gt[flags_gt[j]]))
            else:
                flagsGTNew.append(-1)
        cm = numpy.zeros((len(classes_all), len(classes_all)))
        flags_ind_gt = numpy.array(flagsGTNew)
        for i in range(min(flags_ind.shape[0], flags_ind_gt.shape[0])):
            cm[int(flags_ind_gt[i]), int(flags_ind[i])] += 1
    else:
        flags_ind_gt = numpy.array([])
    acc = plotSegmentationResults(flags_ind, flags_ind_gt, classes_all,
                                  mt_step, not plot_res)
    if acc >= 0:
        # print("Overall Accuracy: {0:.2f}".format(acc))
        return (flags_ind, class_names_gt, acc, cm)
    else:
        return (flags_ind, classes_all, -1, -1)


def trainHMM_computeStatistics(features, labels):
    '''
    This function computes the statistics used to train an HMM joint segmentation-classification model
    using a sequence of sequential features and respective labels

    ARGUMENTS:
     - features:    a numpy matrix of feature vectors (numOfDimensions x n_wins)
     - labels:    a numpy array of class indices (n_wins x 1)
    RETURNS:
     - start_prob:    matrix of prior class probabilities (n_classes x 1)
     - transmat:    transition matrix (n_classes x n_classes)
     - means:    means matrix (numOfDimensions x 1)
     - cov:        deviation matrix (numOfDimensions x 1)
    '''
    u_labels = numpy.unique(labels)
    n_comps = len(u_labels)

    n_feats = features.shape[0]

    if features.shape[1] < labels.shape[0]:
        print("trainHMM warning: number of short-term feature vectors "
              "must be greater or equal to the labels length!")
        labels = labels[0:features.shape[1]]

    # compute prior probabilities:
    start_prob = numpy.zeros((n_comps,))
    for i, u in enumerate(u_labels):
        start_prob[i] = numpy.count_nonzero(labels == u)
    # normalize prior probabilities
    start_prob = start_prob / start_prob.sum()

    # compute transition matrix:
    transmat = numpy.zeros((n_comps, n_comps))
    for i in range(labels.shape[0]-1):
        transmat[int(labels[i]), int(labels[i + 1])] += 1
    # normalize rows of transition matrix:
    for i in range(n_comps):
        transmat[i, :] /= transmat[i, :].sum()

    means = numpy.zeros((n_comps, n_feats))
    for i in range(n_comps):
        means[i, :] = numpy.matrix(features[:,
                                   numpy.nonzero(labels ==
                                                 u_labels[i])[0]].mean(axis=1))

    cov = numpy.zeros((n_comps, n_feats))
    for i in range(n_comps):
        #cov[i,:,:] = numpy.cov(features[:,numpy.nonzero(labels==u_labels[i])[0]])  # use this lines if HMM using full gaussian distributions are to be used!
        cov[i, :] = numpy.std(features[:, numpy.nonzero(labels ==
                                                        u_labels[i])[0]],
                              axis=1)

    return start_prob, transmat, means, cov


def readSegmentGT(gt_file):
    '''
    This function reads a segmentation ground truth file, following a simple CSV format with the following columns:
    <segment start>,<segment end>,<class label>

    ARGUMENTS:
     - gt_file:       the path of the CSV segment file
    RETURNS:
     - seg_start:     a numpy array of segments' start positions
     - seg_end:       a numpy array of segments' ending positions
     - seg_label:     a list of respective class labels (strings)
    '''
    f = open(gt_file, 'rt')
    reader = csv.reader(f, delimiter=',')
    seg_start = []
    seg_end = []
    seg_label = []
    for row in reader:
        if len(row) == 3:
            seg_start.append(float(row[0]))
            seg_end.append(float(row[1]))
            #if row[2]!="other":
            #    seg_label.append((row[2]))
            #else:
            #    seg_label.append("silence")
            seg_label.append((row[2]))
    return numpy.array(seg_start), numpy.array(seg_end), seg_label


def plotSegmentationResults(flags_ind, flags_ind_gt, class_names, mt_step, ONLY_EVALUATE=False):
    '''
    This function plots statistics on the classification-segmentation results produced either by the fix-sized supervised method or the HMM method.
    It also computes the overall accuracy achieved by the respective method if ground-truth is available.
    '''
    # print(flags_ind)
    # print(flags_ind_gt)
    # print(class_names)
    # print(mt_step)
    flags = [class_names[int(f)] for f in flags_ind]
    (segs, classes) = flags2segs(flags, mt_step)
    min_len = min(flags_ind.shape[0], flags_ind_gt.shape[0])
    if min_len > 0:
        accuracy = numpy.sum(flags_ind[0:min_len] ==
                             flags_ind_gt[0:min_len]) / float(min_len)
    else:
        accuracy = -1

    if not ONLY_EVALUATE:
        duration = segs[-1, 1]
        s_percentages = numpy.zeros((len(class_names), 1))
        percentages = numpy.zeros((len(class_names), 1))
        av_durations = numpy.zeros((len(class_names), 1))

        for iSeg in range(segs.shape[0]):
            s_percentages[class_names.index(classes[iSeg])] += \
                (segs[iSeg, 1]-segs[iSeg, 0])

        for i in range(s_percentages.shape[0]):
            percentages[i] = 100.0 * s_percentages[i] / duration
            S = sum(1 for c in classes if c == class_names[i])
            if S > 0:
                av_durations[i] = s_percentages[i] / S
            else:
                av_durations[i] = 0.0

        for i in range(percentages.shape[0]):
            print(class_names[i], percentages[i], av_durations[i])

        font = {'size': 10}
        plt.rc('font', **font)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_yticks(numpy.array(range(len(class_names))))
        ax1.axis((0, duration, -1, len(class_names)))
        ax1.set_yticklabels(class_names)
        ax1.plot(numpy.array(range(len(flags_ind))) * mt_step +
                 mt_step / 2.0, flags_ind)
        if flags_ind_gt.shape[0] > 0:
            ax1.plot(numpy.array(range(len(flags_ind_gt))) * mt_step +
                     mt_step / 2.0, flags_ind_gt + 0.05, '--r')
        plt.xlabel("time (seconds)")
        if accuracy >= 0:
            plt.title('Accuracy = {0:.1f}%'.format(100.0 * accuracy))

        ax2 = fig.add_subplot(223)
        plt.title("Classes percentage durations")
        ax2.axis((0, len(class_names) + 1, 0, 100))
        ax2.set_xticks(numpy.array(range(len(class_names) + 1)))
        ax2.set_xticklabels([" "] + class_names)
        #ax2.bar(numpy.array(range(len(class_names))) + 0.5, percentages)

        ax3 = fig.add_subplot(224)
        plt.title("Segment average duration per class")
        ax3.axis((0, len(class_names) + 1, 0, av_durations.max()))
        ax3.set_xticks(numpy.array(range(len(class_names) + 1)))
        ax3.set_xticklabels([" "] + class_names)
        #ax3.bar(numpy.array(range(len(class_names))) + 0.5, av_durations)
        fig.tight_layout()
        plt.show()
    return accuracy


def flags2segs(flags, window):
    '''
    ARGUMENTS:
     - flags:      a sequence of class flags (per time window)
     - window:     window duration (in seconds)

    RETURNS:
     - segs:       a sequence of segment's limits: segs[i,0] is start and
                   segs[i,1] are start and end point of segment i
     - classes:    a sequence of class flags: class[i] is the class ID of
                   the i-th segment
    '''

    preFlag = 0
    cur_flag = 0
    n_segs = 0

    cur_val = flags[cur_flag]
    segsList = []
    classes = []
    while (cur_flag < len(flags) - 1):
        stop = 0
        preFlag = cur_flag
        preVal = cur_val
        while (stop == 0):
            cur_flag = cur_flag + 1
            tempVal = flags[cur_flag]
            if ((tempVal != cur_val) | (cur_flag == len(flags) - 1)):  # stop
                n_segs = n_segs + 1
                stop = 1
                cur_seg = cur_val
                cur_val = flags[cur_flag]
                segsList.append((cur_flag * window))
                classes.append(preVal)
    segs = numpy.zeros((len(segsList), 2))

    for i in range(len(segsList)):
        if i > 0:
            segs[i, 0] = segsList[i-1]
        segs[i, 1] = segsList[i]
    return (segs, classes)


def segs2flags(seg_start, seg_end, seg_label, win_size):
    '''
    This function converts segment endpoints and respective segment
    labels to fix-sized class labels.
    ARGUMENTS:
     - seg_start:    segment start points (in seconds)
     - seg_end:    segment endpoints (in seconds)
     - seg_label:    segment labels
      - win_size:    fix-sized window (in seconds)
    RETURNS:
     - flags:    numpy array of class indices
     - class_names:    list of classnames (strings)
    '''
    flags = []
    class_names = list(set(seg_label))
    curPos = win_size / 2.0
    while curPos < seg_end[-1]:
        for i in range(len(seg_start)):
            if curPos > seg_start[i] and curPos <= seg_end[i]:
                break
        flags.append(class_names.index(seg_label[i]))
        curPos += win_size
    return numpy.array(flags), class_names
# END SEGMENTATION SECTION


# START IO SECTION
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
        if extension.lower() == '.mp3' or extension.lower() == '.wav' or extension.lower() == '.au' or extension.lower() == '.ogg':
            try:
                audiofile = AudioSegment.from_file(path)
            except:
                print("Error: file not found or other I/O error. "
                      "(DECODING FAILED)")
                return -1, -1

            if audiofile.sample_width == 2:
                data = numpy.fromstring(audiofile._data, numpy.int16)
            elif audiofile.sample_width == 4:
                data = numpy.fromstring(audiofile._data, numpy.int32)
            else:
                return -1, -1
            Fs = audiofile.frame_rate
            x = numpy.array(data[0::audiofile.channels]).T
        else:
            print("Error in readAudioFile(): Unknown file type!")
            return -1, -1
    except IOError:
        print("Error: file not found or other I/O error.")
        return -1, -1

    return Fs, x
# END IO SECTION
