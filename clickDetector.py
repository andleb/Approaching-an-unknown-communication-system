import itertools as it

import numpy as np
from scipy import signal

nCat = 5

fs = 32000
ythresh = 0.4
tthresh = int(0.04 * fs)
btfltr = signal.butter(5, [2000, 15999], btype='bandpass', analog=False, output='sos', fs=fs)


def preprocess(filtered):
    return (filtered - np.mean(filtered)) / np.max(filtered)


def clickCounter(data):
    processed = preprocess(signal.sosfilt(btfltr, data))

    # local maxima
    diff = np.diff(processed) > 0

    dd = diff[:-1] & ~diff[1:] & (processed[1:-1] > ythresh)

    # Enforce temporal separation
    return (np.diff(np.where(dd)) > tthresh).sum() + 1


def clickDetector(data, fltr=True):
    processed = preprocess(signal.sosfilt(btfltr, data) if fltr else data)

    # local maxima
    diff = np.diff(processed) > 0

    dd = diff[:-1] & ~diff[1:] & (processed[1:-1] > ythresh)

    # Enforce temporal separation
    if dd.sum() == 0:
        return [], 0
    elif dd.sum() == 1:
        return np.where(dd)[0] + 1, 1
    else:
        candPeaks = np.where(dd)[0] + 1
        assert ~(processed[candPeaks] < ythresh).any()
        spacings = (np.diff(candPeaks) > tthresh).astype(int)

        # good ones
        peaks = []
        # two-peak situation
        if len(spacings) == 1 and spacings[0]:
            peaks = candPeaks.copy()
            return np.array(sorted(peaks), dtype=int), len(peaks)
        else:
            for i in range(len(spacings) - 1):
                if spacings[i] and spacings[i + 1]:
                    peaks.append(candPeaks[i + 1])

        # bracketing peaks always ok
        if spacings[0] and candPeaks[0] not in peaks:
            peaks.append(candPeaks[0])
        #  not checked above
        if spacings[-1]:
            peaks.append(candPeaks[-1])

        # suspect ones: within each grouping, go in order of magnitude and test if OK
        groups = []
        i = 0
        for k, g in it.groupby(spacings, lambda x: x == 0):
            inext = i + len(list(g)) + 1
            groups.append([k, (i, inext)])
            i = inext - 1

        groups = np.array(groups, dtype=object)
        groups = groups[groups[:, 0].astype(bool)]

        # last good peak before unclear groups
        prevPeak = None
        if len(peaks) and len(groups):
            prevPeaks = [peak for peak in peaks if peak < candPeaks[groups[0][1][0]]]
            prevPeak = prevPeaks[-1] if len(prevPeaks) else None

        for ig, group in enumerate(groups):

            # next peak - next good peak
            try:
                indNextPeak = np.where(spacings[group[1][1] - 1:])[0][0] + group[1][1]
                if spacings[indNextPeak]:
                    # isolated good peak
                    nextPeak = candPeaks[indNextPeak]
                else:
                    # best case: take furthest away among next unclear group
                    # will discard the whole current group if too close
                    nextPeak = candPeaks[groups[ig + 1][1][1] - 1]
            except IndexError:
                nextPeak = None

            cands = candPeaks[range(*group[1])]
            ampls = np.array(sorted([(c, a) for c, a in zip(cands, processed[cands])],
                                    key=lambda t: t[1], reverse=True))
            assert (ampls > ythresh).all()

            # go in order of magnitude and check if spaced enough
            # when successive dubious groupings, will go in order
            currPeaks = []
            for cand in ampls:

                # bracketing accepted peaks
                prevPeakGs = [peak for peak in currPeaks if peak < cand[0]] + ([prevPeak] if prevPeak else [])
                prevPeakG = max(prevPeakGs) if len(prevPeakGs) else None

                nextPeakGs = [peak for peak in currPeaks if peak > cand[0]] + ([nextPeak] if nextPeak else [])
                nextPeakG = min(nextPeakGs) if len(nextPeakGs) else None

                if (((cand[0] - prevPeakG) > tthresh) if prevPeakG else True) and \
                        (((nextPeakG - cand[0]) > tthresh) if nextPeakG else True):
                    currPeaks.append(cand[0])

            peaks += currPeaks

        return np.array(sorted(peaks), dtype=int), len(peaks)
