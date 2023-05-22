import gc
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import torch
from scipy import signal as sp

from clickDetector import clickDetector
from infowavegan import WaveGANGenerator

np.seterr(divide='ignore')

# AUDIO PROPERTIES
fs = 32000
fltr = sp.butter(N=5, Wn=2500, btype='highpass', analog=False, output='sos', fs=fs)

sl = fs * 0.025


def weighted_mean(var, wts):
    """Calculates the weighted mean"""
    return np.average(var, weights=wts)


def weighted_variance(var, wts):
    """Calculates the weighted variance"""
    return np.average((var - weighted_mean(var, wts)) ** 2, weights=wts)


def weighted_std(var, wts):
    """Calculates the weighted standard dev"""
    return np.sqrt(weighted_variance(var, wts))


def weighted_skew(var, wts):
    """Calculates the weighted skewness"""
    return (np.average((var - weighted_mean(var, wts)) ** 3, weights=wts) /
            weighted_variance(var, wts) ** (1.5))


def weighted_kurtosis(var, wts):
    """Calculates the weighted skewness"""
    return (np.average((var - weighted_mean(var, wts)) ** 4, weights=wts) /
            weighted_variance(var, wts) ** (2))


def getNoiseThresh(Sxx):
    logmean = np.log(Sxx).mean(axis=0)
    logmean2 = logmean[np.isfinite(logmean)]
    hh = np.histogram(logmean2, bins=Sxx.shape[1] // 10)

    # find the kink in the distribution
    iKink = np.argmax(np.diff(hh[0]))
    thresh = np.exp(hh[1][iKink])

    return thresh


def getFilteredSpectrogram(genDataF, fs=fs):
    f, t, Sxx = sp.spectrogram(genDataF, fs)

    ind = Sxx > getNoiseThresh(Sxx)
    freqs = np.tile(np.array(f), (len(t), 1)).T[ind]

    return freqs, Sxx[ind]


def filterAndCalc(data):
    """ calculate stats for unit i out of N per bit:t level"""

    # replace in-place
    data = sp.sosfilt(fltr, data)

    peaks, nclicks = clickDetector(data, filter=False)

    ff, codaSpectrum = sp.periodogram(data, fs=fs, window="hamming")

    freqs, Sxx = getFilteredSpectrogram(data)
    clicks = [data[int(ici - sl): int(ici + sl)] for ici in peaks]

    # results
    codaStats = [weighted_mean(freqs, Sxx),
                 weighted_std(freqs, Sxx),
                 weighted_skew(freqs, Sxx),
                 weighted_kurtosis(freqs, Sxx),
                 ]

    clickMeans = []
    clickStds = []
    clickSkews = []
    clickKurtoses = []

    clickSpectra = []

    for click in clicks:

        # NOTE: this fails sometimes when noisy - ie t < 0
        try:
            ffC, PP = sp.periodogram(click, fs=fs, window="hamming")

            clickMeans.append(weighted_mean(ffC, PP))
            clickStds.append(weighted_std(ffC, PP))
            clickSkews.append(weighted_skew(ffC, PP))
            clickKurtoses.append(weighted_kurtosis(ffC, PP))

            clickSpectra.append(PP)

        except ZeroDivisionError:
            continue

    clickStats = [np.mean(clickMeans),
                  np.std(clickMeans),
                  [clickMeans, clickStds,
                   clickSkews, clickKurtoses]
                  ]

    # NOTE: don't normalize yet!
    return nclicks, codaStats, clickStats, (ff, codaSpectrum), (ffC, clickSpectra)


def averagePerLevel(res):
    pass


if __name__ == "__main__":

    genPath = sys.argv[1]
    print(sys.argv)
    print(genPath)

    # PREAMBLE
    # NOTE: use all cpus here
    nCPU = int(mp.cpu_count())
    chunksize = 200

    print("nCPU: ", nCPU, "chunksize: ", chunksize)
    print("affinity: ", len(os.sched_getaffinity(0)))

    nCat = 5
    Nts = 20
    N = 2500

    print("Param lengths: ", nCat, Nts, N)

    bits = range(nCat)
    # NOTE: below this pure noise
    ts = np.linspace(-0.9, 12.5, Nts)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device used :", device)

    # NOTE: make sure all batches are the same size for averaging!
    batchSize = N
    print("Using gpu batches of size ", batchSize)

    Gi = WaveGANGenerator(slice_len=2 ** 16)
    weights = torch.load(genPath, map_location='cpu')
    Gi.load_state_dict(weights)
    G = torch.nn.DataParallel(Gi)
    G.to(f'cuda:{G.device_ids[0]}')

    _z = torch.FloatTensor(N, 100 - nCat).uniform_(-1, 1).to(device)
    c = torch.FloatTensor(N, nCat).to(device)
    z = torch.cat((c, _z), dim=1)
    z[:, :nCat] = 0

    print("Opening pool with", nCPU, "CPUs")
    pool = mp.Pool(nCPU)

    averageData = []
    wholeData = []

    for bit in bits:
        print(f"Generating for bits {bit}, N={N}, {len(ts)} values of t...")

        z[:, :nCat] = 0

        for t in ts:
            print(f"Generating for bit={bit}, N={N}, t={t}")
            z[:, bit] = t

            # Workaround due to GPU memory limits
            # list of promises
            spectrogramData = []

            for zPart in z.split(batchSize, dim=0):
                # 1st async level - filter
                spectrogramData.append(pool.map_async(filterAndCalc,
                                                      G(zPart).detach().cpu().flatten(start_dim=1, end_dim=2).numpy(),
                                                      chunksize=chunksize)
                                       )

            # 2nd level - averaging
            if len(spectrogramData) <= 1:
                dat = spectrogramData[0].get()

                wholeData.append((bit, t, [
                    [row[1][0] for row in dat],  # mean CODA freqs
                    [row[2][0] for row in dat],  # mean CLICK mean freqs-
                    [row[2][1] for row in dat],  # std of mean CLICK freqs
                ]))

                averageData.append((bit, t, [
                    np.nanmean([row[0] for row in dat], axis=0),  # mean click#
                    np.nanmean([row[1] for row in dat], axis=0),  # mean CODA stats over units (Z)
                    np.nanmean([row[2][:2] for row in dat], axis=0),
                    # mean CLICK stats - only on the click mean freq & mean freq std
                    np.nanmean([row[3] for row in dat], axis=0),  # mean CODA spectrum over units: freqs:spectrum
                ]))

                del dat
                del spectrogramData
                gc.collect()

            else:
                chunkAvgData = []
                chunkWholeData = []

                for chunk in spectrogramData:
                    dat = chunk.get()

                    chunkWholeData.append([
                        [row[1][0] for row in dat],  # mean CODA freqs
                        [row[2][0] for row in dat],  # mean CLICK mean freqs-
                        [row[2][1] for row in dat],  # std of mean CLICK freqs
                    ])

                    chunkAvgData.append([
                        np.nanmean([row[0] for row in dat]),
                        np.nanmean(np.array([row[1] for row in dat]), axis=0),
                        np.nanmean(np.array([row[2][:2] for row in dat]), axis=0),
                        np.nanmean(np.array([row[3] for row in dat]), axis=0)
                    ]
                    )

                    del dat
                    gc.collect()

                del spectrogramData
                gc.collect()

                # NOTE: average together the chunks - assuming ~equal chunk sizes!
                wholeData.append((bit, t, np.concatenate([chunk[0] for chunk in chunkWholeData]),
                                  np.concatenate([chunk[1] for chunk in chunkWholeData]),
                                  np.concatenate([chunk[2] for chunk in chunkWholeData])
                                  ))

                averageData.append((bit, t, [
                    np.nanmean([chunk[0] for chunk in chunkAvgData], axis=0),
                    np.nanmean([chunk[1] for chunk in chunkAvgData], axis=0),
                    np.nanmean([chunk[2] for chunk in chunkAvgData], axis=0),
                    np.nanmean([chunk[3] for chunk in chunkAvgData], axis=0),
                ]))

                del chunkWholeData, chunkAvgData
                gc.collect()

            print(f"Generated for bit={bit}, N={N}, t={t}")

        print(f"Generated for bit={bit}, N={N}, {len(ts)} values of t...")

    pool.close()
    pool.join()

    print("Saving")
    np.save("data/audio" + genPath[7:-5].replace("/", "") + "Ts.npy",
            ts)
    np.save("data/audio" + genPath[7:-5].replace("/", "") + "Zs.npy",
            z[:, nCat:].detach().cpu())

    with open("data/audioAvg" + genPath[7:-5].replace("/", "") + ".npy", 'wb') as f:
        pickle.dump(averageData, f)

    with open("data/audioWhole" + genPath[7:-5].replace("/", "") + ".npy", 'wb') as f:
        pickle.dump(wholeData, f)
